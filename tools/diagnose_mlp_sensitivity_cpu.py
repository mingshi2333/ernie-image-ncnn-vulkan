#!/usr/bin/env python3
"""Small-token mathematical MLP reconstruction; not a full official execution oracle."""
import json,math,struct,sys,hashlib,gc
from pathlib import Path
import numpy as np
def erf(x):
 return np.fromiter((math.erf(float(v)) for v in x.flat),dtype=x.dtype,count=x.size).reshape(x.shape)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def run(root,out):
 root=root.resolve();out=out.resolve()
 if out.exists():raise ValueError('Use fresh output')
 out.mkdir(parents=True)
 b75=root/'outputs/q2-block15-boundary-v1';b88=root/'outputs/q2-block15-boundary88-v1';p=json.loads((b75/'plan.json').read_text())
 read=lambda f:np.memmap(f,'<f4','r').reshape(4160,4096)
 a={s:read(b75/s/'boundary.f32') for s in ['official','native']};b={s:read(b88/s/'boundary.f32') for s in a}
 fixture=Path(p['sides']['official']['fixture']);shift=np.fromfile(fixture/'in4.f32','<f4');scale=np.fromfile(fixture/'in5.f32','<f4');gate=np.fromfile(fixture/'in6.f32','<f4')
 scores=[]
 for i in range(4160):
  d=(b['native'][i].astype('f8')-b['official'][i].astype('f8'))*gate;scores.append(float(d@d))
 rows=sorted({0,4095,4096,4159,int(np.argmax(scores))});x=np.concatenate([np.array(a[s][rows]) for s in a]);observed=np.concatenate([np.array(b[s][rows]) for s in a])
 package=root/'models/turbo1024-s64-portable/manifest.json';manifest=json.loads(package.read_text());file=root/'models/official/dit-block-15.safetensors'
 assert sha(package)=='72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1'
 assert sha(file)==manifest['source_weights']['dit'][15]=='d3c1b748148895cf9f4a72669bd77d69a1315c024bcc4f1e5f4f96ecd283a3b2'
 with file.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));offset=8+n
 def weight(name,dtype):
  h=header['layers.15.'+name];assert h['dtype']=='BF16';v=np.memmap(file,'<u2','r',offset=offset+h['data_offsets'][0],shape=tuple(h['shape']));return (v.astype('u4')<<16).view('f4').astype(dtype)
 values={}
 for dt in [np.float32,np.float64]:
  z=x.astype(dt);norm=weight('adaLN_mlp_ln.weight',dt);z=z/np.sqrt(np.mean(z*z,axis=-1,keepdims=True)+dt(1e-6));z=z*norm;z=z*(dt(1)+scale.astype(dt))+shift.astype(dt)
  w=weight('mlp.up_proj.weight',dt);up=z@w.T;del w;gc.collect()
  w=weight('mlp.gate_proj.weight',dt);g=z@w.T;raw_g=g.copy();del w;gc.collect();g=dt(.5)*g*(dt(1)+erf(g/dt(math.sqrt(2))));hidden=up*g
  w=weight('mlp.linear_fc2.weight',dt);y=hidden@w.T
  if dt is np.float64:
   zabs=np.abs(raw_g)/math.sqrt(2);t=1/(1+.3275911*zabs);tail=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*np.exp(-zabs*zabs)
   approx=raw_g*np.where(raw_g>0,1-.5*tail,.5*tail);values['float64_native_gelu_approx']=(up*approx)@w.T
  del w;gc.collect();values[str(np.dtype(dt))]=y
  y.tofile(out/(str(np.dtype(dt))+'.bin'))
 def stats(d):return {'l2':float(np.linalg.norm(d)),'max':float(abs(d).max())}
 k=len(rows);native_delta=observed[k:].astype('f8')-observed[:k].astype('f8');delta64=values['float64'][k:]-values['float64'][:k];delta32=values['float32'][k:].astype('f8')-values['float32'][:k].astype('f8')
 source=next((root/'.venv/lib').glob('python*/site-packages/diffusers/models/transformers/transformer_ernie_image.py'))
 result={'scope':'Selected-row independent mathematical reconstruction, not official CUDA execution or complete tensor acceptance','rows':rows,'selection':'maximum saved gate-weighted down-output conditional difference row plus image/text endpoints','source_sha256':sha(source),'weights_sha256':sha(file),'script_sha256':sha(__file__),'native_vs_fp64_native_gelu_approx':stats(observed.astype('f8')-values['float64_native_gelu_approx']),'intrinsic_gelu_approx_vs_fp64':stats(values['float64_native_gelu_approx']-values['float64']),'gelu_approx_native_error_cosine':float(np.sum((observed.astype('f8')-values['float64'])*(values['float64_native_gelu_approx']-values['float64']))/(np.linalg.norm(observed.astype('f8')-values['float64'])*np.linalg.norm(values['float64_native_gelu_approx']-values['float64']))),'native_vs_fp64':stats(observed.astype('f8')-values['float64']),'numpy_fp32_vs_fp64':stats(values['float32'].astype('f8')-values['float64']),'native_conditional_delta':stats(native_delta),'fp64_conditional_delta':stats(delta64),'native_delta_minus_fp64_delta':stats(native_delta-delta64),'numpy_fp32_delta_minus_fp64_delta':stats(delta32-delta64)}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':run(Path.cwd(),Path(sys.argv[1]))
