import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base/'source/tools'))
from pipeline_reference import full_reference_contract
from pipeline_package import select_shared_instance
request=json.loads((base/'request.json').read_text())
plan=json.loads((base/'plan.json').read_text())
native=base
ref=base/'official/reference'
def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for name,item in plan['bindings'].items():
 path=Path(name)
 assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
assert json.loads((base/'official/process.json').read_text())['complete']
assert json.loads((native/'native/process.json').read_text())['complete']
fixture=json.loads((ref/'fixture.json').read_text())
source=json.loads((Path(request['package'])/'manifest.json').read_text())
expected_config={**source['config'],'packed_width':request['runtime_size'][0]//16,'packed_height':request['runtime_size'][1]//16}
assert full_reference_contract(fixture,expected_config,request['prompt'],request['steps'])==25
assert fixture['ids']==[int(v) for v in (native/'native/trace/ids.txt').read_text().split()]
assert fixture['runtime_source']['source_manifest_sha256']==sha(Path(request['package'])/'manifest.json')
native_plan=json.loads((native/'plan.json').read_text())
for name,item in native_plan['bindings'].items():
 path=Path(name)
 assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
command=native_plan['commands']['native'][0]
shared=next(Path(name).parent for name,item in native_plan['bindings'].items() if Path(name).name=='manifest.json' and item['sha256']==native_plan['model_manifest_sha256'])
assert sha(shared/'manifest.json')==native_plan['model_manifest_sha256']
shared_manifest=json.loads((shared/'manifest.json').read_text())
selected,_=select_shared_instance(shared_manifest['instances'],*request['runtime_size'],len(fixture['ids']))
assert selected['config']==expected_config
assert selected['source_manifest_sha256']==fixture['runtime_source']['source_manifest_sha256']
entries=[(name,item,True) for name,item in fixture['inputs'].items()]
for i,outputs in enumerate(fixture['outputs']):entries.extend((f'{name}-{i}',item,False) for name,item in outputs.items())
entries.extend((name,item,False) for name,item in fixture['final'].items())
rows=[]
for name,item,conditioning in entries:
 expected_path=ref/item['file'];actual_path=native/'native/trace'/(name+'.f32')
 assert sha(expected_path)==item['sha256'],name
 expected=np.fromfile(expected_path,'<f4').astype('f8');actual=np.fromfile(actual_path,'<f4').astype('f8')
 assert expected.size==actual.size==int(np.prod(item['shape'])) and np.isfinite(expected).all() and np.isfinite(actual).all(),name
 delta=actual-expected
 error={'max_abs_error':float(np.abs(delta).max()),'nrmse':float(np.sqrt(np.dot(delta,delta)/max(np.dot(expected,expected),1e-30))),'reference_max_abs':float(np.abs(expected).max())}
 gate=plan['gates']['conditioning' if conditioning else 'fp32']
 passed=error['nrmse']<=gate['nrmse'] and error['max_abs_error']<=gate['atol']+gate['global_rtol']*error['reference_max_abs']
 if name=='initial':passed=np.array_equal(actual,expected)
 rows.append({'name':name,'elements':actual.size,**error,'passed':bool(passed),'reference_sha256':sha(expected_path),'native_sha256':sha(actual_path)})
assert len(rows)==25
png=ref/'reference.png';assert sha(png)==fixture['reference_png_sha256']
a=np.asarray(Image.open(native/'native/native.png').convert('RGB')).astype('f8')
b=np.asarray(Image.open(png).convert('RGB')).astype('f8')
assert a.shape==b.shape==(request['runtime_size'][1],request['runtime_size'][0],3)
delta=np.abs(a-b);gate=plan['gates']['fp32']
pixel={'mae':float(delta.mean()),'max_abs':float(delta.max()),'different_channel_samples':int(np.count_nonzero(delta)),'total_channel_samples':int(delta.size),'reference_sha256':sha(png),'native_sha256':sha(native/'native/native.png')}
pixel['passed']=pixel['mae']<=gate['pixel_mae'] and pixel['max_abs']<=gate['pixel_max']
for directory,pixels in ((native/'native/trace',a),(ref,b)):
 decoded=np.fromfile(directory/'decoded.f32','<f4').reshape(3,*a.shape[:2])
 quantized=(np.clip(decoded/2+.5,0,1).transpose(1,2,0)*255).round().astype('uint8')
 assert np.array_equal(pixels,quantized)
result={'scope':'Native text and 8-step runtime1376x768 trajectory with explicit RAM weights/stdio/cache0 with source hidden/network disabled versus staged official FP32 modules using identical saved initial latent; official CUDA blocks/CPU text-heads-VAE, native Vulkan blocks/CPU text-VAE; CPU2 and trace on; not performance or broad perceptual acceptance',
 'reference_fixture_sha256':sha(ref/'fixture.json'),'native_run':str(native),'runtime_source':fixture['runtime_source'],
 'gates':plan['gates'],'comparisons':rows,'total_elements':sum(x['elements'] for x in rows),'passed_tensors':sum(x['passed'] for x in rows),
 'png':pixel,'both_png_quantizations_exact':True,'passed':bool(pixel['passed'] and all(x['passed'] for x in rows))}
(base/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key] for key in ('total_elements','passed_tensors','png','passed')},indent=2))

raise SystemExit(0 if result["passed"] else 1)
