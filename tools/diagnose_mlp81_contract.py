#!/usr/bin/env python3
"""CPU-only fixed block15 MLP input contract. Never starts a model or GPU."""
import json,struct,sys
from pathlib import Path
from diagnose_official_block_hooks import sha,require
PARAM='e8ed770153823bc60b2f27cfac14b9a58f534eb6bd9f19dc692d650f7faddd13'
BINARY='0321f883d79cab6606ac3d2736344043625f3a0fed9bb1f5fcfdd33d1acd07da'
WEIGHT='d3c1b748148895cf9f4a72669bd77d69a1315c024bcc4f1e5f4f96ecd283a3b2'
BASELINE={'75':'09189225db6307daa124c88dce40dfc257e8af49d9cea125b44c6c64072b69af','87':'384328e7033248ef5c43d1a9494d4f93c7e13556bd0913bdda786cfc1f1f10dd','88':'3a1eb976b4772659e9461745bb21ec9fa92582adfd98d90f728aab83d8375ab8','out0':'a3f77e846934a6fe7085bf366dd21ef78aa779b702c5c97ba237834a0635a197'}
NATIVE='175bfa394aec3b7e576caaea3fd897cf84db122d4cd9378d7d72dfe704df68a3'
SHAPES={'75':[1,4160,4096],'81':[1,4160,4096],'87':[1,4160,12288],'88':[1,4160,4096],'out0':[1,4160,4096]}
def baseline_gate(hashes):
 require(all(hashes.get(k)==v for k,v in BASELINE.items()),'Official baseline did not reproduce all four old complete tensors')
def tail_needed(native_hash,official_hash,native_out):
 require(native_out==NATIVE,'Native observer changed complete teacher output')
 require(all(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in [native_hash,official_hash]),'Invalid 81 hash')
 return native_hash!=official_hash

def proof(root):
 from audit_port_weights import layer_weights,normalized_hash,official_inventory
 model=root/'models/turbo1024-s64-portable/dit/block-15';param=model/'block.ncnn.param';binary=model/'block.ncnn.bin';weight=root/'models/official/dit-block-15.safetensors'
 require(sha(param)==PARAM and sha(binary)==BINARY and sha(weight)==WEIGHT,'Fixed actual model identity')
 rows=[];offset=0
 with binary.open('rb') as f:
  for line in param.read_text().splitlines()[2:]:
   t=line.split();kind,name=t[:2];p=dict(x.split('=',1) for x in t[4+int(t[2])+int(t[3]):])
   if kind in ('SDPA','ErnieResidualAdd','ErnieGELU'):continue # serialized-weight-free, pinned whole graph only
   for role,count,load,transpose in layer_weights(kind,p):
    dtype='F32'
    if load==0:
     f.seek(offset);tag=struct.unpack('<I',f.read(4))[0];offset+=4
     require(tag in (0,0x0002c056,0x01306b47,0x01348b83),'Unknown weight tag')
     if tag==0x01306b47:dtype='F16'
     if tag==0x01348b83:dtype='BF16'
    start=offset;size=count*(2 if dtype in ('F16','BF16') else 4);offset+=(size+3)//4*4
    if name in ('rmsn_11','gemm_4','gemm_5','gemm_6'):
     rows.append(dict(layer=name,role=role,offset=start,count=count,dtype=dtype,transpose=transpose,canonical_sha256=normalized_hash(binary,start,count,dtype,transpose)))
 require(offset==binary.stat().st_size,'Unconsumed native bytes')
 official=official_inventory(root/'models/official',['dit-block-15.safetensors'])
 names={'rmsn_11':'adaLN_mlp_ln.weight','gemm_4':'mlp.up_proj.weight','gemm_5':'mlp.gate_proj.weight','gemm_6':'mlp.linear_fc2.weight'}
 for row in rows:
  target='layers.15.'+names[row['layer']];matches=[r for r in official if r['name']==target]
  require(len(matches)==1 and matches[0]['canonical_sha256']==row['canonical_sha256'],'Logical weight mismatch '+target)
  row.update(official_tensor=target,official_shape=matches[0]['shape'])
 return {'param_sha256':PARAM,'binary_sha256':BINARY,'official_component_sha256':WEIGHT,'consumed_binary_bytes':offset,'weights':rows,'exact_graph_tail':'\n'.join(param.read_text().splitlines()[-21:]),'mapping':'81 = learned RMSNorm(75, eps=1e-6) * (1 + in5 scale_mlp) + in4 shift_mlp; native FP32 row major [4160,4096], official mlp prehook [4160,1,4096]; disk BSH [1,4160,4096]. Split 81 -> 83/82, gemm_4(83)->84 up; gemm_5(82)->85 gate; ErnieGELU(85)->86; BinaryOp mul(84,86)->87; gemm_6(87)->88 down. in6 is the subsequent residual gate, not an MLP input.'}

def prepare(root,out):
 require(not out.exists(),'Existing output');out.mkdir(parents=True)
 result=proof(root)
 old=root/'outputs/q2-block15-official-hooks-v3';oldplan=old/'plan.json'
 require(sha(oldplan)=='f6aac83a83415bf6eda28b244fa6b16219c4633e9eee5b09ca0e69a57e278519','Old oracle plan')
 p=json.loads(oldplan.read_text());bound=dict(p['bound']);bound[str(oldplan.resolve())]=sha(oldplan)
 for name,digest in BASELINE.items():
  f=old/'actual'/f'{name}.f32';require(sha(f)==digest,'Old oracle bytes');bound[str(f.resolve())]=digest
 for f in [root/'tools/diagnose_mlp81_contract.py',root/'tools/audit_port_weights.py',root/'tools/diagnose_official_block_hooks.py']:
  bound[str(f.resolve())]=sha(f)
 result.update(scope='diagnostic only; matched MLP input; no production/formal gate',status='cpu_contract_prepared_gpu_not_run',shapes=SHAPES,native_expected_out0=NATIVE,official_baseline_expected=BASELINE,max_forwards={'native_full_block':1,'official_full_block':1,'official_mlp_only':1},skip_tail_if_81_byte_equal=True,baseline_prerequisite='native out0 exact; official out0+75+87+88 exact before accepting 81 or running tail',matched_input_rule='Only block.mlp actual input is replaced with complete native81, disk BSH transposed to SBH; do not replace raw75, weights, float32/backend/TF32 settings or conditioning.',decomposition={'propagation':'official_mlp_pre_down(native81) - official87','matched_implementation':'native87 - official_mlp_pre_down(native81)'},bound=bound,formal_acceptance=False)
 (out/'contract.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'contract_sha256':sha(out/'contract.json'),'weights':result['weights']}))
if __name__=='__main__':prepare(Path.cwd().resolve(),Path(sys.argv[1]).resolve())
