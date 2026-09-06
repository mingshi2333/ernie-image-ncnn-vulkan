#!/usr/bin/env python3
"""Complete matched-input decomposition; CPU only, no inferred official values."""
import json,sys
from pathlib import Path
import numpy as np
from diagnose_official_block_hooks import sha,verify,require
from diagnose_mlp81_contract import BASELINE,baseline_gate
RESULT='f62eb1d8ae79579ea135afc016290a14143ef4661c8dfccc701a36c28a855b54'

def split_vectors(native,official,matched):
 require(native.shape==official.shape==matched.shape and all(a.dtype==np.dtype('float32') for a in [native,official,matched]),'Matched inputs require exact equal shapes and FP32')
 d=native.astype('f8')-official;u=matched.astype('f8')-official;e=native.astype('f8')-matched
 require(np.allclose(d,u+e,rtol=0,atol=1e-12),'Decomposition identity failed')
 return d,u,e

def compare(a,b):
 sums=ref=0.;maximum=0.;count=0;changed=0;rows=[]
 for i in range(0,a.shape[0],64):
  x=a[i:i+64].astype('f8');y=b[i:i+64].astype('f8');d=x-y
  sums+=float(np.sum(d*d));ref+=float(np.sum(y*y));maximum=max(maximum,float(np.max(np.abs(d))));count+=d.size;changed+=int(np.count_nonzero(d));rows.extend(np.sum(d*d,axis=1).tolist())
 return {'elements':count,'l2':sums**.5,'max_abs':maximum,'nrmse':(sums/ref)**.5 if ref else None,'changed_elements':changed,'top_l2_rows':np.argsort(rows)[-8:][::-1].tolist()}

def decomposition(n,o,m):
 sums=np.zeros(3,'f8');dots=0.;maxima=np.zeros(3,'f8');row=[]
 for i in range(0,n.shape[0],64):
  vectors=split_vectors(n[i:i+64],o[i:i+64],m[i:i+64])
  for j,v in enumerate(vectors):sums[j]+=np.sum(v*v);maxima[j]=max(maxima[j],np.max(np.abs(v)))
  dots+=float(np.sum(vectors[1]*vectors[2]));row.extend(np.sum(vectors[2]**2,axis=1).tolist())
 return {'elements':int(n.size),'total':{'l2':float(sums[0]**.5),'max_abs':float(maxima[0])},'propagated_input':{'l2':float(sums[1]**.5),'max_abs':float(maxima[1])},'matched_implementation':{'l2':float(sums[2]**.5),'max_abs':float(maxima[2])},'propagation_local_cosine':float(dots/(sums[1]*sums[2])**.5) if sums[1]*sums[2] else None,'complete_sum_identity_atol':1e-12,'largest_local_rows':np.argsort(row)[-8:][::-1].tolist()}

def run(root,destination):
 out=root/'outputs/q2-block15-mlp81-official-v5';require(not destination.exists(),'New analysis path required')
 require(sha(out/'result.json')==RESULT,'Execution result identity');r=json.loads((out/'result.json').read_text());p=json.loads((out/'plan.json').read_text());verify(p['bound'])
 require(r['forwards']=={'native_full_block':0,'official_full_block':1,'official_mlp_only':1},'Forward denominator');baseline_gate(r['baseline_sha256'])
 identities={};arrays={}
 def read(label,path,digest,width=4096):
  require(sha(path)==digest and path.stat().st_size==4160*width*4,'Wrong input '+label)
  a=np.memmap(path,dtype='<f4',mode='r',shape=(4160,width));require(np.isfinite(a).all(),'Nonfinite '+label);identities[str(path)]=digest;arrays[label]=a
 for key in ['75','81','87','88','out0']:read('O'+key,out/'official'/f'{key}.f32',r['baseline_sha256'][key],12288 if key=='87' else 4096)
 for key in ['87','88']:read('M'+key,out/'official'/f'native81-{key}.f32',r['tail_sha256'][key],12288 if key=='87' else 4096)
 read('N81',out/'native/boundary.f32','da8bc040080f6674042c860a372c8f004041dae617a0cdeffae96f3c2be61d54')
 read('Nout0',out/'native/out0.f32','175bfa394aec3b7e576caaea3fd897cf84db122d4cd9378d7d72dfe704df68a3')
 for key,folder,digest in [('75','q2-block15-boundary-v1','762941a013ce27957c6b123985eb7ef1dd36fafe7c79c7f08cb7569ac154c1bd'),('87','q2-block15-boundary87-v1','a951b1ed772d5a53c0b30ba8a1d1882c52491f6c564cb9f70b09ec6ef4cd3c46'),('88','q2-block15-boundary88-v1','b1d26a4acc2c44d7864c25ba59c821ded6fb8e8c34575d7b35d0ca9586b95cbf')]:
  read('N'+key,root/'outputs'/folder/'official/boundary.f32',digest,12288 if key=='87' else 4096)
 result={'scope':'Complete real tensor same-input MLP and input-propagation decomposition; diagnostic only','execution_result_sha256':RESULT,'analysis_source_sha256':sha(__file__),'identity':identities,'comparisons':{key:compare(arrays['N'+key],arrays['O'+key]) for key in ['75','81','87','88','out0']},'decomposition':{key:decomposition(arrays['N'+key],arrays['O'+key],arrays['M'+key]) for key in ['87','88']},'definitions':{'N':'native on exact official block14 teacher input','O':'official baseline on same teacher input','M':'official MLP only, actual input replaced with entire native81; for 87 capture pre-down, for 88 module output','U':'M-O: propagation of 81 difference using official implementation','E':'N-M: matched-input implementation difference; 88 includes all upstream MLP differences, not isolated down rounding'},'formal_acceptance':False}
 destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['comparisons','decomposition']},indent=2))
if __name__=='__main__':run(Path.cwd().resolve(),Path(sys.argv[1]).resolve())
