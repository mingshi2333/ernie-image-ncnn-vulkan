#!/usr/bin/env python3
"""Fixed one-layer candidate, never a pipeline acceptance protocol."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from diagnose_splitk_screen import sha,metrics
PARAM_SHA='e8ed770153823bc60b2f27cfac14b9a58f534eb6bd9f19dc692d650f7faddd13'
BIN_SHA='0321f883d79cab6606ac3d2736344043625f3a0fed9bb1f5fcfdd33d1acd07da'

def rewrite_graph(data):
 if hashlib.sha256(data).hexdigest()!=PARAM_SHA:raise ValueError('Unreviewed graph')
 lines=data.splitlines(keepends=True);indices=[i for i,s in enumerate(lines) if s.split()[:2]==[b'Gemm',b'gemm_6']]
 if len(indices)!=1:raise ValueError('Expected one down projection')
 i=indices[0]
 if lines[i].split()[2:]!=b'1 1 87 88 10=-1 2=0 3=1 4=0 5=1 6=1 7=4160 8=4096 9=12288'.split():raise ValueError('Down projection contract mismatch')
 lines[i]=lines[i].replace(b'Gemm',b'Q2SplitDown',1)
 return b''.join(lines)

def chunks(rows=4160,size=16):
 if type(rows)!=int or type(size)!=int or rows<=0 or size<=0:raise ValueError('Invalid row plan')
 return [(s,min(size,rows-s)) for s in range(0,rows,size)]

def complete(path,width):
 if Path(path).stat().st_size!=4160*width*4:raise ValueError('Incomplete tensor '+str(path))
 a=np.memmap(path,'<f4','r',shape=(4160,width))
 if not np.isfinite(a).all():raise ValueError('Nonfinite tensor')
 return a

def prepare(root,out):
 old=root/'outputs/q2-block15-boundary87-v1';base=json.loads((old/'plan.json').read_text());bound=base['bound'].copy()
 screen_dir=root/'outputs/q2-down-splitk-v3'
 screen_plan=json.loads((screen_dir/'plan.json').read_text());screen_result=json.loads((screen_dir/'result.json').read_text())
 if screen_result['passed'] is not True or screen_result['actual_sha256']!='f8e56eacfb0a53b58567557a12c832094b8b09ebffb6b74866c587bb04b41808' or sha(screen_dir/'actual.f32')!=screen_result['actual_sha256']:raise ValueError('Screen prerequisite identity')
 bound.update(screen_plan['bound'])
 for f,h in bound.items():
  if sha(f)!=h:raise ValueError('Historical source changed '+f)
 teacher=json.loads((root/'outputs/q2-chinese-step0-block15-official-input-v1/plan.json').read_text())
 model=Path(base['model']);dest=out/'model';dest.mkdir()
 if sha(model/'block.ncnn.bin')!=BIN_SHA:raise ValueError('Unreviewed weights')
 (dest/'block.ncnn.param').write_bytes(rewrite_graph((model/'block.ncnn.param').read_bytes()))
 os.link(model/'block.ncnn.bin',dest/'block.ncnn.bin')
 fixture=out/'fixture';fixture.mkdir()
 for f in Path(base['sides']['official']['fixture']).glob('in*.f32'):os.link(f,fixture/f.name)
 fields={'baseline':old/'official/out0.f32','upstream':old/'official/boundary.f32','residual':root/'outputs/q2-block15-boundary-v1/official/boundary.f32','oracle':Path(teacher['expected']['file']),'screen':root/'outputs/q2-down-splitk-v3/actual.f32','screen_weight':root/'outputs/q2-down-splitk-v3/fixture/weight-kn.f32'}
 for file in [old/'plan.json',root/'outputs/q2-down-splitk-v3/plan.json',root/'outputs/q2-down-splitk-v3/result.json',*fields.values(),*fixture.iterdir(),*dest.iterdir(),*(out/'execution').rglob('*')]:
  if file.is_file():bound[str(file)]=sha(file)
 p={'scope':'Single-block official-input teacher; diagnostic only, no formal gate','output':str(out),'fields':{k:str(v) for k,v in fields.items()},'bound':bound,'chunks':chunks(),'elements':4160*4096,'rows':[0,22,4095,4096,4159],'upstream_must_be_bitwise_equal':True,'screen_selected_rows_must_be_bitwise_equal':True,'residual_reconstruction_must_be_bitwise_equal':True,'original_graph_sha256':PARAM_SHA,'candidate_layer':'gemm_6 only: 87 -> 88','original_weight_sha256':BIN_SHA}
 (out/'plan.json').write_text(json.dumps(p,indent=2)+'\n');return p

def execute(path):
 p=json.loads(path.read_text());out=Path(p['output']);e=out/'execution'
 for f,h in p['bound'].items():
  if sha(f)!=h:raise ValueError('Changed frozen identity '+f)
 if p['chunks']!=[list(v) for v in chunks()] or p['elements']!=4160*4096:raise ValueError('Wrong complete denominator')
 subprocess.run([str(e/'runner'),str(out/'model'),str(out/'fixture'),str(out/'candidate'),'88',str(e/'shaders'),p['fields']['screen_weight']],check=True)
 candidate=out/'candidate';fields=p['fields']
 complete(candidate/'upstream87.f32',12288)
 if sha(candidate/'upstream87.f32')!=sha(fields['upstream']):raise ValueError('Upstream changed: candidate attribution invalid')
 down=complete(candidate/'boundary.f32',4096);screen=np.fromfile(fields['screen'],'<f4').reshape(10,4096)
 if not np.array_equal(down[p['rows']].view('u4'),screen[:5].view('u4')):raise ValueError('Tiled math does not reproduce screened selected rows')
 final=complete(candidate/'out0.f32',4096);residual=complete(fields['residual'],4096);gate=np.fromfile(out/'fixture/in6.f32','<f4')
 reconstruction=(residual+(down*gate).astype('f4')).astype('f4')
 if not np.array_equal(final.view('u4'),reconstruction.view('u4')):raise ValueError('Final residual/layout mismatch')
 oracle=complete(fields['oracle'],4096);baseline=complete(fields['baseline'],4096)
 result={'scope':p['scope'],'status':'valid_complete_teacher','plan_sha256':sha(path),'complete_elements':final.size,'upstream_bitwise_equal':True,'screen_selected_rows_bitwise_equal':True,'residual_reconstruction_bitwise_equal':True,'baseline_vs_official':metrics(baseline,oracle),'candidate_vs_official':metrics(final,oracle),'candidate_vs_baseline':metrics(final,baseline),'outputs':{str(f):sha(f) for f in candidate.iterdir() if f.is_file()},'formal_acceptance':False,'quality_gate_applied':False}
 for f,h in p['bound'].items():
  if sha(f)!=h:raise ValueError('Changed frozen identity after execution '+f)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare(Path.cwd().resolve(),Path(sys.argv[2]).resolve())
 else:execute(Path(sys.argv[2]))
