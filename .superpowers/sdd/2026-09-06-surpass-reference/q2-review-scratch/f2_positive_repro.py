import hashlib,json,sys,tempfile,shutil,types
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'tools'))
from reference_img2img_positive import validate_inputs,run
source=Path('outputs/f2-positive05-512x384-v1/inputs');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)/'inputs';shutil.copytree(source,root)
 original=json.loads((root/'input-contract.json').read_text())
 for mode in ['arbitrary_start','wrong_encoder_bytes','wrong_sigma_and_bn']:
  c=json.loads(json.dumps(original))
  if mode=='arbitrary_start':
   x=np.zeros((1,128,24,32),dtype='<f4');x.tofile(root/'start-4.f32');c['start']['sha256']=sha(root/'start-4.f32')
  elif mode=='wrong_encoder_bytes':(root/'out2.f32').write_bytes(bytes((root/'out2.f32').stat().st_size))
  else:c['request']['sigma']=.123;c['encoder_fixture']['encoder_bn_eps']=1;c['sigmas_f32']=[0]
  (root/'input-contract.json').write_text(json.dumps(c));validate_inputs(root);print(mode,'ACCEPTED')
  shutil.copyfile(source/'start-4.f32',root/'start-4.f32');shutil.copyfile(source/'out2.f32',root/'out2.f32')
 (root/'input-contract.json').write_text(json.dumps(original))
 def fake_reference(package,prompt,out,steps,device,start,start_step):
  out.mkdir();fixture={'complete':True,'steps':2,'start_step':0,'outputs':[],'final':{},'prompt':'wrong'}
  (out/'fixture.json').write_text(json.dumps(fixture));return fixture
 sys.modules['validate_pipeline']=types.SimpleNamespace(reference=fake_reference)
 result=run(Path(tmp)/'nonexistent-package',root,Path(tmp)/'result','cpu')
 print('empty_wrong_suffix_complete',result['complete'])
# Actual artifact lightweight verification: no model or torch import.
c=json.loads((source/'input-contract.json').read_text());enc=json.loads((source/'fixture.json').read_text())
for item in [enc['rgb'],*enc['expected'].values()]:assert sha(source/item['file'])==item['sha256']
sigma=np.float32(.800000011920929)
start=sigma*np.fromfile(source/'saved-noise.f32','<f4')+(np.float32(1)-sigma)*np.fromfile(source/'out2.f32','<f4')
assert start.tobytes()==(source/'start-4.f32').read_bytes()
out=Path('outputs/f2-positive05-512x384-v1/official');r=json.loads((out/'reference.json').read_text());f=json.loads((out/'suffix/fixture.json').read_text())
assert sha(out/'suffix/fixture.json')==r['suffix_fixture_sha256']
assert f==r['suffix'] and f['steps']==8 and f['start_step']==4 and len(f['outputs'])==4
entries=list(f['inputs'].values())+list(f['final'].values())
for i,record in enumerate(f['outputs'],4):
 for key in ['step','prediction']:assert record[key]['file']==f'{key}-{i}.f32';entries.append(record[key])
for item in entries:
 path=out/'suffix'/item['file'];assert sha(path)==item['sha256'];assert path.stat().st_size==int(np.prod(item['shape']))*4
assert sha(out/'suffix/reference.png')==f['reference_png_sha256']
print('actual_encoder_start_and_suffix',len(entries),'tensor hashes/sizes and PNG verified; actual data not invalidated')
