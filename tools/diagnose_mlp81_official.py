#!/usr/bin/env python3
"""Fixed official baseline then at most one same-input MLP call; diagnostic only."""
import json,sys
from pathlib import Path
import numpy as np
from diagnose_official_block_hooks import sha,require,verify,imports,freeze_runtime,tensor_bytes
from diagnose_mlp81_contract import BASELINE,NATIVE,SHAPES,baseline_gate,tail_needed

def run(plan_path):
 p=json.loads(plan_path.read_text());verify(p['bound']);out=Path(p['output']);old=Path(p['old_official'])
 n=out/'native';require(sha(n/'out0.f32')==NATIVE,'Native final changed; no official execution')
 require((n/'boundary.f32').stat().st_size==4160*4096*4,'Incomplete native81')
 torch,_,load_block=imports(out)
 def read(file,shape):
  require(file.stat().st_size==int(np.prod(shape))*4,'Incomplete tensor')
  a=np.fromfile(file,'<f4');require(np.isfinite(a).all(),'Nonfinite input');return torch.from_numpy(a).reshape(shape).to('cuda')
 f=old/'fixture';x=read(f/'in0.f32',(1,4160,4096)).transpose(0,1)
 temb=[read(f/f'in{i}.f32',(1,1,4096)) for i in range(1,7)]
 angles=read(f/'angles.f32',(1,4160,1,128));mask=read(f/'in9.f32',(1,1,4160,4160))
 block,component=load_block(Path(p['weight']),15);block=block.to('cuda');require(component['sha256']==p['official_component_sha256'],'Weight changed')
 require(not block.training and torch.get_float32_matmul_precision()=='highest' and block.self_attention.processor._attention_backend is None,'Backend changed')
 config={'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),'device':torch.cuda.get_device_name(),'threads':torch.get_num_threads(),'attention_processor':type(block.self_attention.processor).__name__,'attention_backend':str(block.self_attention.processor._attention_backend),'training':block.training,'config':torch.__config__.show()}
 require(config==json.loads((old/'execution/torch-config.json').read_text()),'Actual runtime options changed')
 (out/'execution/torch-config.json').write_text(json.dumps(config,indent=2)+'\n')
 before=freeze_runtime(out/'execution','gpu-before-forward-runtime')
 captures={};counts={k:0 for k in ['75','81','87','88']}
 def capture(key,value):
  counts[key]+=1;require(counts[key]==1,'Duplicate baseline hook');captures[key]=value.detach()
 handles=[block.adaLN_mlp_ln.register_forward_pre_hook(lambda m,a:capture('75',a[0])),block.mlp.register_forward_pre_hook(lambda m,a:capture('81',a[0])),block.mlp.linear_fc2.register_forward_pre_hook(lambda m,a:capture('87',a[0])),block.mlp.linear_fc2.register_forward_hook(lambda m,a,y:capture('88',y))]
 final=block(x,angles,temb,attention_mask=mask);torch.cuda.synchronize()
 for h in handles:h.remove()
 require(all(v==1 for v in counts.values()),'Missing baseline hook')
 actual=out/'official';actual.mkdir();hashes={}
 for name,value in [*captures.items(),('out0',final)]:
  path=actual/(name+'.f32');path.write_bytes(tensor_bytes(value.transpose(0,1),SHAPES[name]));hashes[name]=sha(path)
 (out/'baseline-observation.json').write_text(json.dumps({'hashes':hashes,'counts':counts},indent=2)+'\n')
 baseline_gate(hashes)
 native81=sha(n/'boundary.f32');needed=tail_needed(native81,hashes['81'],sha(n/'out0.f32'))
 tail_hashes={};tail_counts={'87':0,'88':0}
 if needed:
  # The entire *post-conditioning* native81 is the sole substituted input.
  # Linear module, weight storage, backend and FP32 dtype remain unchanged.
  captures.clear();del final,x,temb,angles,mask
  inp=read(n/'boundary.f32',(1,4160,4096)).transpose(0,1)
  def tail_capture(key,value):
   tail_counts[key]+=1;require(tail_counts[key]==1,'Duplicate MLP hook');captures[key]=value.detach()
  h1=block.mlp.linear_fc2.register_forward_pre_hook(lambda m,a:tail_capture('87',a[0]))
  h2=block.mlp.linear_fc2.register_forward_hook(lambda m,a,y:tail_capture('88',y))
  tail=block.mlp(inp);torch.cuda.synchronize();h1.remove();h2.remove()
  require(tail_counts=={'87':1,'88':1},'Missing MLP hook')
  for name,value in captures.items():
   path=actual/('native81-'+name+'.f32');path.write_bytes(tensor_bytes(value.transpose(0,1),SHAPES[name]));tail_hashes[name]=sha(path)
  require(tail.shape==(4160,1,4096),'Wrong MLP output')
 freeze_runtime(out/'execution','gpu-after-forward-runtime');verify(p['bound']);verify({k:v['sha256'] for k,v in before['files'].items()})
 baseline_gate({k:sha(actual/(k+'.f32')) for k in BASELINE})
 result={'scope':'Matched-input MLP diagnostic; not official acceptance','status':'completed','plan_sha256':sha(plan_path),'baseline_sha256':hashes,'native81_sha256':native81,'tail_sha256':tail_hashes,'forwards':{'native_full_block':1,'official_full_block':1,'official_mlp_only':int(needed)},'baseline_hook_counts':counts,'tail_hook_counts':tail_counts,'tail_skipped_81_exact':not needed,'shapes':SHAPES,'formal_acceptance':False}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]))
