#!/usr/bin/env python3
"""CPU-only decomposition after whole official out0 identity, never inferred oracle tensors."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from diagnose_official_block_hooks import EXPECTED,SHAPES,sha,require

def read(file,width):
 require(Path(file).stat().st_size==4160*width*4,'Incomplete full boundary')
 a=np.memmap(file,'<f4','r',shape=(4160,width));require(np.isfinite(a).all(),'Nonfinite boundary');return a

def stats(a):return {'l2':float(np.linalg.norm(a)),'max':float(abs(a).max())}
def metric(a,b):
 total=0.;maximum=0.;ref=0.
 for start in range(0,len(a),64):
  aa=a[start:start+64].astype('f8');bb=b[start:start+64].astype('f8');d=aa-bb
  total+=float(np.sum(d*d));ref+=float(np.sum(bb*bb));maximum=max(maximum,float(abs(d).max()))
 return {'l2':total**.5,'max':maximum,'nrmse':(total/max(ref,1e-300))**.5}
def cosine(a,b):return float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def authenticate_baselines(root):
 teacher=root/'outputs/q2-block15-splitk-teacher-v1'
 require(sha(teacher/'plan.json')=='6bdfa3d9a4d117afa621152e01cf48e4e576bd93dd559306ffea31c7a0805369','Teacher plan identity')
 require(sha(teacher/'result.json')=='3ecaa8354d67eac3de9f7a151b0b94c34f569026545d3e482e8732030587f940','Teacher execution identity')
 plan=json.loads((teacher/'plan.json').read_text());bound={Path(k).resolve():v for k,v in plan['bound'].items()};files={}
 for rel in ['q2-block15-boundary-v1/official/boundary.f32','q2-block15-boundary87-v1/official/boundary.f32','q2-block15-boundary88-v1/official/boundary.f32','q2-block15-boundary87-v1/official/out0.f32']:
  path=(root/'outputs'/rel).resolve();require(path in bound and sha(path)==bound[path],'Historical boundary identity');files[str(path)]=bound[path]
 record=json.loads((teacher/'result.json').read_text())
 for file,digest in record['outputs'].items():require(sha(file)==digest,'Candidate output identity');files[file]=digest
 return files

def run(root,out):
 identity=authenticate_baselines(root)
 r=json.loads((out/'result.json').read_text());require(r['out0_bitwise_equal'] and r['output_sha256']['out0']==EXPECTED,'Unavailable official internal oracle');require(json.loads((out/'execution/worker-result.json').read_text())['passed'],'Official execution incomplete')
 official={name:read(out/'actual'/(name+'.f32'),SHAPES[name][-1]) for name in SHAPES}
 for name in SHAPES:require(sha(out/'actual'/(name+'.f32'))==r['output_sha256'][name],'Official boundary changed')
 native={'75':read(root/'outputs/q2-block15-boundary-v1/official/boundary.f32',4096),'87':read(root/'outputs/q2-block15-boundary87-v1/official/boundary.f32',12288),'88':read(root/'outputs/q2-block15-boundary88-v1/official/boundary.f32',4096),'out0':read(root/'outputs/q2-block15-boundary87-v1/official/out0.f32',4096)}
 candidate={'88':read(root/'outputs/q2-block15-splitk-teacher-v1/candidate/boundary.f32',4096),'out0':read(root/'outputs/q2-block15-splitk-teacher-v1/candidate/out0.f32',4096)}
 full={name:metric(native[name],official[name]) for name in SHAPES};full_candidate={name:metric(candidate[name],official[name]) for name in candidate}
 gate=np.fromfile(out/'fixture/in6.f32','<f4');o75=official['75'];o88=official['88']
 totals={s:{k:0. for k in ['residual_sq','update_sq','rounding_sq','residual_update_dot','out_sq']} for s in ['native','candidate']}
 for start in range(0,4160,64):
  sl=slice(start,start+64);op=(o88[sl]*gate).astype('f4');ob=(o75[sl]+op).astype('f4')
  require(np.array_equal(ob.view('u4'),official['out0'][sl].view('u4')),'Official residual reconstruction differs')
  for side,values in [('native',native),('candidate',candidate)]:
   npart=(values['88'][sl]*gate).astype('f4');nb=(native['75'][sl]+npart).astype('f4');require(np.array_equal(nb.view('u4'),values['out0'][sl].view('u4')),'Native/candidate reconstruction differs')
   residual=native['75'][sl].astype('f8')-o75[sl];update=npart.astype('f8')-op;delta=values['out0'][sl].astype('f8')-official['out0'][sl];rounding=delta-residual-update;t=totals[side]
   for key,a in [('residual_sq',residual),('update_sq',update),('rounding_sq',rounding),('out_sq',delta)]:t[key]+=float(np.sum(a*a))
   t['residual_update_dot']+=float(np.sum(residual*update))
 decomposition={s:{'residual_l2':t['residual_sq']**.5,'gated_update_l2':t['update_sq']**.5,'final_add_rounding_l2':t['rounding_sq']**.5,'out_l2':t['out_sq']**.5,'residual_update_cosine':t['residual_update_dot']/(t['residual_sq']*t['update_sq'])**.5} for s,t in totals.items()}
 # Post-result selected rows, now using authentic official and native pre-down values.
 rows=[0,22,652,4095,4096,4159];weight=root/'outputs/q2-down-splitk-v3/fixture/weight-kn.f32';require(sha(weight)=='e180b28bea1d6dead7c978199d83d4fa2f6886370d4d83f238816d4bfd78f6af','Weight identity')
 w=np.memmap(weight,'<f4','r',shape=(12288,4096));x=np.concatenate([official['87'][rows],native['87'][rows]]).astype('f8');precise=x@w.astype('f8');fo,fn=precise[:6],precise[6:]
 upstream=fn-fo;nl=native['88'][rows].astype('f8')-fn;ol=official['88'][rows].astype('f8')-fo;cl=candidate['88'][rows].astype('f8')-fn
 ne=native['88'][rows].astype('f8')-official['88'][rows];ce=candidate['88'][rows].astype('f8')-official['88'][rows]
 require(np.allclose(upstream+nl-ol,ne,rtol=0,atol=1e-12),'Native decomposition algebra');require(np.allclose(upstream+cl-ol,ce,rtol=0,atol=1e-12),'Candidate decomposition algebra')
 selected={'scope':'Post-result 6 rows, 24576 outputs; FP64 mathematical dot, not another GPU reference execution','rows':rows,'upstream_propagated':stats(upstream),'native_local_dot':stats(nl),'official_local_dot':stats(ol),'candidate_local_dot':stats(cl),'native_vs_official_down':stats(ne),'candidate_vs_official_down':stats(ce),'native_local_vs_official_local_cosine':cosine(nl,ol),'upstream_vs_native_local_cosine':cosine(upstream,nl),'upstream_vs_official_local_cosine':cosine(upstream,ol),'by_row':[]}
 for i,row in enumerate(rows):selected['by_row'].append({'row':row,'upstream':stats(upstream[i]),'native_local':stats(nl[i]),'official_local':stats(ol[i]),'candidate_local':stats(cl[i]),'native_total':stats(ne[i]),'candidate_total':stats(ce[i]),'upstream_native_local_cosine':cosine(upstream[i],nl[i]),'upstream_official_local_cosine':cosine(upstream[i],ol[i])})
 result={'scope':'Authenticated full internal boundaries plus explicitly selected FP64 dot decomposition; not full trajectory acceptance','official_out0_sha256':EXPECTED,'full_native_vs_official':full,'full_candidate_vs_official':full_candidate,'all_three_residuals_bitwise_reconstructed':True,'full_output_decomposition':decomposition,'selected_fp64_decomposition':selected,'analysis_source_sha256':sha(__file__),'historical_baseline_identity':identity,'official_result_sha256':sha(out/'result.json'),'official_worker_sha256':sha(out/'execution/worker-result.json')}
 (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':run(Path.cwd().resolve(),Path(sys.argv[1]).resolve())
