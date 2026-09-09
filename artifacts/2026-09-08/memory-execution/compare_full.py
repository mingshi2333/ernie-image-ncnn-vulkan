"""Recompute native-to-native and official-oracle errors for completed runs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

base = Path(__file__).resolve().parent
plan = json.loads((base/'full-plan.json').read_text())
ref = Path(plan['reference'])
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
fixture=json.loads((ref/'fixture.json').read_text())
assert sha(ref/'fixture.json')==plan['bindings'][str(ref/'fixture.json')]['sha256']
entries=[(name,record,True) for name,record in fixture['inputs'].items()]
for step, records in enumerate(fixture['outputs']):
    entries.extend((f'{name}-{step}',record,False) for name,record in records.items())
entries.extend((name,record,False) for name,record in fixture['final'].items())
assert len(entries)==25 and fixture['complete']
official_image=np.array(Image.open(ref/'reference.png').convert('RGB'))
results=[]
for version,precision in plan['cases']:
    case=version+'-'+precision
    out=base/'full'/case
    if not (out/'process.json').exists():continue
    process=json.loads((out/'process.json').read_text())
    if not process['complete']:continue
    report=json.loads((out/'generation.json').read_text())
    assert report['status']=='success' and report['shape']==[512,512]
    assert report['token_ids']==plan['token_ids'] and report['prompt']==plan['prompt']
    assert report['request']['precision']==precision and report['request']['steps']==8
    assert report['request']['text_down_vector'] and report['request']['dit_weights']=='host'
    assert report['pe']['enabled'] is False and report['trace_enabled'] is True
    assert report['model']=={'schema_version':3,'source_width':1024,'source_height':1024,'text_bucket':32,'dit_text_tokens':64}
    assert [(p['current'],p['total']) for p in report['progress'] if p['stage']=='denoise']==[(i,8) for i in range(1,9)]
    tensors=[]
    for name,record,conditioning in entries:
        expected_path,actual_path=ref/record['file'],out/'trace'/(name+'.f32')
        assert sha(expected_path)==record['sha256']
        a,b=np.fromfile(expected_path,'<f4'),np.fromfile(actual_path,'<f4')
        assert a.shape==b.shape and a.size==int(np.prod(record['shape']))
        assert np.isfinite(a).all() and np.isfinite(b).all()
        delta=b.astype('float64')-a.astype('float64')
        norm=float(np.linalg.norm(delta)/max(np.linalg.norm(a.astype('float64')),1e-30))
        maximum=float(np.max(np.abs(delta))); refmax=float(np.max(np.abs(a)))
        gate=plan['gates']['conditioning' if conditioning else precision]
        limit=gate['atol']+gate['global_rtol']*refmax
        passed=norm<=gate['nrmse'] and maximum<=limit
        if name=='initial':passed=sha(expected_path)==sha(actual_path)
        tensors.append({'name':name,'elements':int(a.size),'finite':True,'nrmse':norm,'max_abs_error':maximum,
            'reference_max_abs':refmax,'max_abs_limit':limit,'nrmse_limit':gate['nrmse'],'passed':passed,
            'reference_sha256':sha(expected_path),'native_sha256':sha(actual_path)})
    image=np.array(Image.open(out/'native.png').convert('RGB'))
    assert image.shape==official_image.shape==(512,512,3)
    delta=np.abs(image.astype('int16')-official_image.astype('int16'))
    gate=plan['gates'][precision]
    decoded=np.fromfile(out/'trace/decoded.f32','<f4').reshape(3,512,512)
    quantized=(np.clip(decoded/2+.5,0,1).transpose(1,2,0)*255).round().astype('uint8')
    assert np.array_equal(quantized,image)
    png={'mae':float(delta.mean()),'max_abs':int(delta.max()),'different_channels':int(np.count_nonzero(delta)),
        'passed':bool(delta.mean()<=gate['pixel_mae'] and delta.max()<=gate['pixel_max']),
        'sha256':sha(out/'native.png'),'quantization_exact':True}
    result={'case':case,'scope':'512x512 / 8 steps / native text / Vulkan DiT / CPU VAE',
        'tensors':tensors,'passed_tensors':sum(r['passed'] for r in tensors),'total_tensors':len(tensors),
        'total_elements':sum(r['elements'] for r in tensors),'png':png,
        'passed':all(r['passed'] for r in tensors) and png['passed'],'process':process}
    before=Path(plan['saved_fp32_baseline']) if precision=='fp32' else Path(plan['saved_bf16_baseline'])
    if (before/'native.png').exists():
        if precision=='fp32':
            anchor=json.loads((base/'historical-fp32-anchor.json').read_text())
            assert sha(before/'native.png')==anchor['png_sha256']
            for name,digest in anchor['tensors'].items():
                assert sha(before/'trace'/name)==digest
        assert {p.name for p in (before/'trace').glob('*.f32')}=={p.name for p in (out/'trace').glob('*.f32')}
        diffs=[]
        for name,_,_ in entries:
            a,b=np.fromfile(before/'trace'/(name+'.f32'),'<f4'),np.fromfile(out/'trace'/(name+'.f32'),'<f4')
            assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            diffs.append({'name':name,'bitwise_equal':sha(before/'trace'/(name+'.f32'))==sha(out/'trace'/(name+'.f32')),
                'max_abs_difference':float(np.max(np.abs(a.astype('float64')-b.astype('float64'))))})
        old_image=np.array(Image.open(before/'native.png'))
        result['old_new']={'baseline':str(before),'tensors':diffs,'bitwise_equal_tensors':sum(r['bitwise_equal'] for r in diffs),
            'png_bitwise_equal':sha(before/'native.png')==sha(out/'native.png'),
            'png_pixels_equal':bool(np.array_equal(old_image,image)),'png_max_abs_difference':int(np.max(np.abs(old_image.astype('int16')-image.astype('int16'))))}
    results.append(result)
(base/'full-comparison.json').write_text(json.dumps({'scope':'Memory execution comparison and unchanged official gates; no speed claim',
    'plan_sha256':sha(base/'full-plan.json'),'comparator_sha256':sha(__file__),'results':results},indent=2)+'\n')
for r in results:
    print(r['case'],'official gates',r['passed_tensors'],'/ 25','PNG MAE/max',r['png']['mae'],r['png']['max_abs'],
          'old/new exact tensors',r.get('old_new',{}).get('bitwise_equal_tensors'),flush=True)
