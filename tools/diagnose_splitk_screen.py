#!/usr/bin/env python3
"""Frozen local screen. Selected rows are explicit; never a formal quality gate."""
import json,hashlib,struct,subprocess,sys
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def metrics(a,b):
 d=a.astype('f8')-b;return {'l2':float(np.linalg.norm(d)),'max':float(abs(d).max())}

def prepare(root,out):
 fixture=out/'fixture';fixture.mkdir();rows=[0,22,4095,4096,4159];b87=root/'outputs/q2-block15-boundary87-v1';b88=root/'outputs/q2-block15-boundary88-v1';bound={}
 for d in [b87,b88]:
  result=json.loads((d/'result.json').read_text());bound[str(d/'result.json')]=sha(d/'result.json')
  for side,rec in zip(['official','native'],result['out0_checks']):
   assert sha(d/side/'boundary.f32')==rec['boundary_sha256'];assert rec['out0_bitwise_equal'];bound[str(d/side/'boundary.f32')]=rec['boundary_sha256']
 x=np.concatenate([np.array(np.memmap(b87/s/'boundary.f32','<f4','r').reshape(4160,12288)[rows]) for s in ['official','native']]);baseline=np.concatenate([np.array(np.memmap(b88/s/'boundary.f32','<f4','r').reshape(4160,4096)[rows]) for s in ['official','native']])
 file=root/'models/official/dit-block-15.safetensors';assert sha(file)=='d3c1b748148895cf9f4a72669bd77d69a1315c024bcc4f1e5f4f96ecd283a3b2';bound[str(file)]=sha(file)
 with file.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));offset=8+n
 e=header['layers.15.mlp.linear_fc2.weight'];assert e['dtype']=='BF16' and e['shape']==[4096,12288]
 w=(np.memmap(file,'<u2','r',offset=offset+e['data_offsets'][0],shape=(4096,12288)).astype('u4')<<16).view('f4');oracle=x.astype('f8')@w.T.astype('f8');w.T.copy().tofile(fixture/'weight-kn.f32');x.tofile(fixture/'input.f32');baseline.tofile(fixture/'baseline.f32');oracle.tofile(fixture/'oracle.f64');del w
 # Independent cancellation/range cases, all exact powers-of-two-scaled integer dots.
 sw=np.zeros((12288,4096),dtype='f4');sw[[0,1,2],0]=[1e8,1,-1e8];sw[[0,1,2],1]=[2**40,1,-2**40];sw[:,2]=np.where(np.arange(12288)%2,1,-1);sw[:,3]=-1;sw[[0,383,384,12287],4]=[1e8,1,-1e8,2]
 sx=np.stack([np.ones(12288,dtype='f4'),np.where(np.arange(12288)%2,.5,-.5).astype('f4'),np.full(12288,2**-20,dtype='f4'),np.full(12288,2**20,dtype='f4')]);so=sx.astype('f8')@sw.astype('f8');sw.tofile(fixture/'synthetic-weight.f32');sx.tofile(fixture/'synthetic-input.f32');so.tofile(fixture/'synthetic-oracle.f64')
 for f in [*fixture.iterdir(),*(out/'execution').rglob('*')]:
  if f.is_file():bound[str(f)]=sha(f)
 plan={'rows':rows,'sides':['official','native'],'M':10,'N':4096,'K':12288,'splits':32,'required_reduction_factor':2,'synthetic_criterion':'all outputs exactly equal rounded FP64 oracle','output':str(out),'bound':bound,'scope':'local diagnostic candidate screen; no formal gate changes'};(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');return plan

def execute(path):
 p=json.loads(path.read_text());out=Path(p['output']);f=out/'fixture';ex=out/'execution'
 for file,digest in p['bound'].items():
  if sha(file)!=digest:raise ValueError('Changed frozen bytes '+file)
 for name,m,x,w in [('actual',10,'input.f32','weight-kn.f32'),('synthetic',4,'synthetic-input.f32','synthetic-weight.f32')]:subprocess.run([str(ex/'runner'),str(f/x),str(f/w),str(m),str(ex/'shaders'),str(out/(name+'.f32'))],check=True)
 a=np.fromfile(out/'actual.f32','<f4').reshape(10,4096);b=np.fromfile(f/'baseline.f32','<f4').reshape(a.shape);o=np.fromfile(f/'oracle.f64','<f8').reshape(a.shape);assert np.isfinite(a).all();rows=[]
 for i,s in enumerate(p['sides']):
  sl=slice(i*5,(i+1)*5);baseline=metrics(b[sl],o[sl]);candidate=metrics(a[sl],o[sl]);rows.append({'side':s,'baseline':baseline,'candidate':candidate,'passed':all(candidate[k]<=baseline[k]/2 for k in ['l2','max'])})
 syn=np.fromfile(out/'synthetic.f32','<f4').reshape(4,4096);so=np.fromfile(f/'synthetic-oracle.f64','<f8').reshape(syn.shape);synthetic=bool(np.isfinite(syn).all() and np.array_equal(syn,so.astype('f4')))
 result={'scope':p['scope'],'plan_sha256':sha(path),'rows':rows,'synthetic':{'passed':synthetic,**metrics(syn,so)},'passed':all(r['passed'] for r in rows) and synthetic,'actual_sha256':sha(out/'actual.f32'),'synthetic_sha256':sha(out/'synthetic.f32'),'formal_acceptance':False};(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare(Path.cwd().resolve(),Path(sys.argv[2]).resolve())
 else:execute(Path(sys.argv[2]))
