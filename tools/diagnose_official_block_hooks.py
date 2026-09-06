#!/usr/bin/env python3
"""One authenticated official block15 forward. No replacement arithmetic or quality gate."""
import hashlib,importlib.metadata,json,os,platform,shutil,sys
from pathlib import Path
PROPOSAL='4d54414723d9b6e6ac0c6fe670404e13aa4f5f2acdc53e61b8ee9d3c006e451e'
EXPECTED='a3f77e846934a6fe7085bf366dd21ef78aa779b702c5c97ba237834a0635a197'
SHAPES={'75':(1,4160,4096),'87':(1,4160,12288),'88':(1,4160,4096),'out0':(1,4160,4096)}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def verify(bound):
 for file,digest in bound.items():require(sha(file)==digest,'Changed bound identity '+file)
def endpoint(actual):
 require(actual==EXPECTED,'Invalid instrumented oracle: complete out0 differs')
def tensor_bytes(value,shape):
 import numpy as np
 a=value.detach().cpu().contiguous().numpy()
 require(tuple(a.shape)==tuple(shape) and a.dtype==np.dtype('float32'),'Wrong full tensor shape/dtype')
 require(np.isfinite(a).all(),'Nonfinite tensor');return a.astype('<f4',copy=False).tobytes()
def freeze_runtime(archive,label):
 files={}
 for name,module in sorted(sys.modules.copy().items()):
  path=getattr(module,'__file__',None)
  if isinstance(path,(str,bytes,os.PathLike)) and Path(path).is_file():files[str(Path(path).resolve())]=name
 for line in Path('/proc/self/maps').read_text().splitlines():
  fields=line.split()
  if len(fields)>=6 and fields[-1].startswith('/') and '.so' in fields[-1] and Path(fields[-1]).is_file():files[str(Path(fields[-1]).resolve())]='mapped_shared_library'
 files[str(Path(sys.executable).resolve())]='python_executable'
 snapshot=archive/'runtime-objects';snapshot.mkdir(exist_ok=True);records={}
 for file,role in files.items():
  path=Path(file);digest=sha(path);target=snapshot/digest
  if not target.exists():
   try:os.link(path,target)
   except OSError:shutil.copyfile(path,target)
  require(sha(target)==digest,'Runtime snapshot changed')
  records[file]={'role':role,'sha256':digest,'bytes':path.stat().st_size,'snapshot':str(target)}
 result={'label':label,'python':sys.version,'platform':platform.platform(),'versions':{name:importlib.metadata.version(name) for name in ['torch','diffusers','numpy','safetensors','transformers']},'files':records,'scope':'Actual imported source paths and mapped runtime libraries authenticated by bytes; external packages execute from those authenticated paths, archive objects retain their bytes.'}
 (archive/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n');return result

def imports(out):
 # Project helper imports execute from the sealed old official scripts, never ROOT/tools.
 sys.path.insert(0,str(out/'execution/imports'))
 import torch
 from export_dit_block import make_inputs,load_block
 import export_dit_block
 require(Path(export_dit_block.__file__).resolve()==(out/'execution/imports/export_dit_block.py').resolve(),'Unsealed helper import')
 torch.set_num_threads(4);torch.set_grad_enabled(False)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 return torch,make_inputs,load_block

def fixed_execution_file(file):
 return file.suffix=='.py' or file.name=='cpu-import-runtime.json' or file.parent.name=='runtime-objects'

def prepare(root,out):
 proposal_path=root/'outputs/q2-block15-splitk-teacher-v1/next-official-hook-proposal.json'
 require(sha(proposal_path)==PROPOSAL,'Proposal identity');proposal=json.loads(proposal_path.read_text());verify(proposal['source_bound'])
 torch,make_inputs,_=imports(out)
 fixture=out/'fixture';fixture.mkdir()
 previous=root/'outputs/q2-block15-splitk-teacher-v1/fixture'
 for file in sorted(previous.glob('in*.f32')):os.link(file,fixture/file.name)
 generated,angles=make_inputs(64,64,64,32,20260905)
 for index,value in zip([7,8,9],generated[7:]):
  raw=tensor_bytes(value,(1,4160,128) if index!=9 else (4160,4160))
  require(hashlib.sha256(raw).hexdigest()==sha(fixture/f'in{index}.f32'),'Generated constant differs '+str(index))
 angles_path=fixture/'angles.f32';angles_path.write_bytes(tensor_bytes(angles,(1,4160,1,128)))
 del generated,angles
 inventory=freeze_runtime(out/'execution','cpu-import-runtime')
 bound=proposal['source_bound'].copy();bound[str(proposal_path)]=PROPOSAL
 for file in [*fixture.iterdir(),*(out/'execution').rglob('*')]:
  if file.is_file() and (file.parent==fixture or fixed_execution_file(file)):bound[str(file)]=sha(file)
 for path,item in inventory['files'].items():bound[path]=item['sha256']
 plan={'scope':'One official block15 with read-only 75/87/88 hooks; out0 bitwise prerequisite','output':str(out),'proposal_sha256':PROPOSAL,'angles_sha256':sha(angles_path),'constants_bitwise_equal':[7,8,9],'bound':bound,'shapes':SHAPES,'weight':str((root/'models/official/dit-block-15.safetensors').resolve()),'expected':EXPECTED,'model_forwards':1,'native_acceptance_eligible':False}
 (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'status':'prepared','constants_bitwise_equal':[7,8,9],'angles_sha256':plan['angles_sha256'],'plan_sha256':sha(out/'plan.json'),'imported_runtime_files':len(inventory['files'])}),flush=True)

def execute(path):
 import numpy as np
 p=json.loads(path.read_text());out=Path(p['output']);verify(p['bound']);require(p['proposal_sha256']==PROPOSAL and p['expected']==EXPECTED and p['model_forwards']==1,'Protocol changed')
 require(p['shapes']=={k:list(v) for k,v in SHAPES.items()},'Output denominator changed')
 torch,_,load_block=imports(out)
 f=out/'fixture'
 def read(name,shape):return torch.from_numpy(np.fromfile(f/name,'<f4').copy()).reshape(shape)
 x=read('in0.f32',(1,4160,4096)).transpose(0,1).to('cuda')
 temb=[read(f'in{i}.f32',(1,1,4096)).to('cuda') for i in range(1,7)]
 mask=read('in9.f32',(4160,4160))[None,None].to('cuda');angles=read('angles.f32',(1,4160,1,128)).to('cuda')
 block,component=load_block(Path(p['weight']),15);require(component['sha256']=='d3c1b748148895cf9f4a72669bd77d69a1315c024bcc4f1e5f4f96ecd283a3b2','Official component changed');block=block.to('cuda')
 (out/'execution/torch-config.json').write_text(json.dumps({'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),'device':torch.cuda.get_device_name(),'threads':torch.get_num_threads(),'attention_processor':type(block.self_attention.processor).__name__,'attention_backend':str(block.self_attention.processor._attention_backend),'training':block.training,'config':torch.__config__.show()},indent=2)+'\n')
 require(torch.get_float32_matmul_precision()=='highest' and block.self_attention.processor._attention_backend is None,'Official math options differ')
 before=freeze_runtime(out/'execution','gpu-before-forward-runtime')
 captured={};counts={key:0 for key in ['75','87','88']}
 def capture(name,value):
  counts[name]+=1;require(counts[name]==1,'Repeated hook '+name)
  # Retain the unmodified value and its storage owner. Official forward uses out-of-place operations.
  captured[name]=value.detach()
 def before_norm(module,args):capture('75',args[0])
 def before_down(module,args):capture('87',args[0])
 def after_down(module,args,value):capture('88',value)
 handles=[block.adaLN_mlp_ln.register_forward_pre_hook(before_norm),block.mlp.linear_fc2.register_forward_pre_hook(before_down),block.mlp.linear_fc2.register_forward_hook(after_down)]
 final=block(x,angles,temb,attention_mask=mask);torch.cuda.synchronize()
 for h in handles:h.remove()
 require(counts=={'75':1,'87':1,'88':1},'Incomplete hooks')
 actual=out/'actual';actual.mkdir();hashes={}
 for name,value in [*captured.items(),('out0',final)]:
  file=actual/(name+'.f32');file.write_bytes(tensor_bytes(value.transpose(0,1),SHAPES[name]));hashes[name]=sha(file)
 result={'scope':p['scope'],'status':'valid_instrumented_official_oracle' if hashes['out0']==EXPECTED else 'invalid_instrumented_oracle','plan_sha256':sha(path),'out0_bitwise_equal':hashes['out0']==EXPECTED,'expected_out0_sha256':EXPECTED,'output_sha256':hashes,'hook_counts':counts,'shapes':SHAPES,'model_forwards':1,'formal_acceptance':False,'intermediate_interpretation_allowed':hashes['out0']==EXPECTED}
 (out/'forward-observation.json').write_text(json.dumps(result,indent=2)+'\n')
 freeze_runtime(out/'execution','gpu-after-forward-runtime')
 verify(p['bound']);verify({file:item['sha256'] for file,item in before['files'].items()})
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2),flush=True);endpoint(hashes['out0'])
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare(Path.cwd().resolve(),Path(sys.argv[2]).resolve())
 else:execute(Path(sys.argv[2]))
