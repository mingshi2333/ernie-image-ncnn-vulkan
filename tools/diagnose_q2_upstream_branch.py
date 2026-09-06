#!/usr/bin/env python3
"""Read-only complete-tensor branch response; no new model execution."""
import json,sys
from pathlib import Path
import numpy as np
from diagnose_official_block_hooks import sha,require

def read(file,width=4096):
 require(Path(file).stat().st_size==4160*width*4,'Incomplete state')
 a=np.memmap(file,'<f4','r',shape=(4160,width));require(np.isfinite(a).all(),'Nonfinite state');return a

def finish(acc):
 out={k:float(v**.5) for k,v in acc['squares'].items()}
 out['cosines']={name:float(value/(acc['squares'][pair[0]]*acc['squares'][pair[1]])**.5) for name,(pair,value) in acc['dots'].items()}
 return out

def run(root,out):
 require(not out.exists(),'Use new output');out.mkdir()
 teacher=root/'outputs/q2-block15-splitk-teacher-v1';require(sha(teacher/'plan.json')=='6bdfa3d9a4d117afa621152e01cf48e4e576bd93dd559306ffea31c7a0805369','Teacher identity')
 plan=json.loads((teacher/'plan.json').read_text());bound={str(Path(k).resolve()):v for k,v in plan['bound'].items()};identity={}
 def checked(path,expected=None,width=4096):
  path=path.resolve();digest=expected or bound.get(str(path));require(digest is not None and sha(path)==digest,'State identity '+str(path));identity[str(path)]=digest;return read(path,width)
 b=root/'outputs/q2-block15-boundary-v1';b87=root/'outputs/q2-block15-boundary87-v1';b88=root/'outputs/q2-block15-boundary88-v1'
 incoming=checked(b/'native-fixture/in0.f32');exact=checked(b/'official-fixture/in0.f32')
 require(sha(b/'result.json')=='0342c5dfb43b6fcc17ee54fe40e84bb3be09aec7f1bd9fe186c2a71099c0f025','Historical pair result')
 t75=checked(b/'official/boundary.f32');s75=checked(b/'native/boundary.f32','3b297b93ebf874ac23789f3ab81b0d590a40457e1d763fba8864bba27bf4887d')
 t87=checked(b87/'official/boundary.f32',width=12288);s87=checked(b87/'native/boundary.f32',width=12288)
 t88=checked(b88/'official/boundary.f32');s88=checked(b88/'native/boundary.f32')
 tout=checked(b87/'official/out0.f32','175bfa394aec3b7e576caaea3fd897cf84db122d4cd9378d7d72dfe704df68a3');sout=checked(b87/'native/out0.f32','55c6e86b1bc14acb1acdcd15383c1055ee8fe46ff744657418435370fdadc5e6')
 gate_path=b/'official-fixture/in6.f32';require(sha(gate_path)==bound[str(gate_path.resolve())],'Gate identity');gate=np.fromfile(gate_path,'<f4');identity[str(gate_path.resolve())]=sha(gate_path)
 names=['incoming_delta','first_residual_delta','attention_branch_response','mlp_branch_response','out_delta','gated_mlp_response','second_add_rounding_response','pre_down_delta','down_delta']
 acc={'squares':{n:0. for n in names},'dots':{'input_attention':(('incoming_delta','attention_branch_response'),0.),'first_residual_mlp':(('first_residual_delta','mlp_branch_response'),0.),'attention_mlp':(('attention_branch_response','mlp_branch_response'),0.)}}
 row_scores={name:[] for name in ['incoming_delta','attention_branch_response','mlp_branch_response','out_delta']}
 for start in range(0,4160,64):
  sl=slice(start,start+64);d0=incoming[sl].astype('f8')-exact[sl];d75=s75[sl].astype('f8')-t75[sl];df=sout[sl].astype('f8')-tout[sl];da=d75-d0;dm=df-d75
  gm=(s88[sl]*gate).astype('f4').astype('f8')-(t88[sl]*gate).astype('f4').astype('f8')
  require(np.array_equal((s75[sl]+(s88[sl]*gate).astype('f4')).astype('f4').view('u4'),sout[sl].view('u4')),'Stack reconstruction')
  require(np.array_equal((t75[sl]+(t88[sl]*gate).astype('f4')).astype('f4').view('u4'),tout[sl].view('u4')),'Teacher reconstruction')
  values=dict(zip(names,[d0,d75,da,dm,df,gm,dm-gm,s87[sl].astype('f8')-t87[sl],s88[sl].astype('f8')-t88[sl]]))
  for name,value in values.items():acc['squares'][name]+=float(np.sum(value*value))
  for name,(pair,total) in list(acc['dots'].items()):acc['dots'][name]=(pair,total+float(np.sum(values[pair[0]]*values[pair[1]])))
  for name in row_scores:row_scores[name].extend(np.sum(values[name]**2,axis=1).tolist())
 result={'scope':'Native block15 conditional response to existing incoming stack-vs-official discrepancy; not local same-input rounding error or global condition number','denominators':{'state':17039360,'pre_down':51118080},'norms':finish(acc),'top_rows':{n:np.argsort(v)[-5:][::-1].tolist() for n,v in row_scores.items()},'definitions':{'attention_branch_response':'(S75-T75)-(S14-O14), including first-add rounding','mlp_branch_response':'(Sout-Tout)-(S75-T75), including second-add rounding','gated_mlp_response':'FP32(gate*S88)-FP32(gate*T88)','S':'native execution on accumulated native block14 input','T':'same native binary on exact official block14 input','O':'official block14 input'},'source_sha256':sha(__file__),'input_identity':identity,'formal_acceptance':False}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':run(Path.cwd().resolve(),Path(sys.argv[1]).resolve())
