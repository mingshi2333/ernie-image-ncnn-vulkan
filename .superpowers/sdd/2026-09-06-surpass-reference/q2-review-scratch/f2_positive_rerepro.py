import hashlib,json,sys,tempfile,shutil,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'tools'))
from reference_img2img_positive import validate_inputs,validate_suffix
source=Path('outputs/f2-positive05-512x384-v1/inputs');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for mode in ['arbitrary_start','wrong_encoder_bytes','wrong_sigma_and_bn','alternate_names']:
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp)/'inputs';shutil.copytree(source,root);c=json.loads((root/'input-contract.json').read_text())
  if mode=='arbitrary_start':
   np.zeros((1,128,24,32),dtype='<f4').tofile(root/'start-4.f32');c['start']['sha256']=sha(root/'start-4.f32')
  elif mode=='wrong_encoder_bytes':(root/'out2.f32').write_bytes(bytes((root/'out2.f32').stat().st_size))
  elif mode=='wrong_sigma_and_bn':c['request']['sigma']=.123;c['encoder_fixture']['encoder_bn_eps']=1;c['sigmas_f32']=[0]
  else:
   noise=np.zeros((1,128,24,32),dtype='<f4');encoded=np.fromfile(root/'out2.f32','<f4').reshape(noise.shape)
   start=np.float32(.8)*noise+(np.float32(1)-np.float32(.8))*encoded
   for key,value in [('noise',noise),('start',start)]:
    file=f'alternate-{key}.f32';value.tofile(root/file);c[key].update(file=file,sha256=sha(root/file))
  (root/'input-contract.json').write_text(json.dumps(c))
  try:validate_inputs(root)
  except ValueError as error:print(mode,'REJECTED',str(error))
  else:print(mode,'ACCEPTED')
real=Path('outputs/f2-positive05-512x384-v1/official-invalid-prompt-trimmed/suffix');fixture=json.loads((real/'fixture.json').read_text());contract,prompt=validate_inputs(source)
try:validate_suffix(real,fixture,prompt,contract['start']['sha256'])
except ValueError as error:print('historical_trimmed_prompt','REJECTED',str(error))
print('historical_structure_only_denominator',validate_suffix(real,fixture,fixture['prompt'],contract['start']['sha256']))
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);bad=copy.deepcopy(fixture);bad['prompt']=prompt;bad['outputs']=[];bad['final']={};(root/'fixture.json').write_text(json.dumps(bad))
 try:validate_suffix(root,bad,prompt,contract['start']['sha256'])
 except ValueError as error:print('empty_suffix','REJECTED',str(error))
 else:print('empty_suffix','ACCEPTED')
