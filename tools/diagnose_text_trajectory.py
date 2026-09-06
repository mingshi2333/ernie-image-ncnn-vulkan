#!/usr/bin/env python3
"""Read-only, identity-bound comparison of saved eight-step text trajectories.

Prediction error on divergent inputs is NOT a local DiT arithmetic error. Euler
residuals here describe the stored update equation, not causal error percentages.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose_trajectory import summarize_run, canonical_sha256


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def finite_arrays(*arrays):
    arrays=[np.asarray(a,dtype=np.float64) for a in arrays]
    if not arrays or not arrays[0].size or any(a.shape!=arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Require nonempty equal finite tensor shapes')
    return arrays


def cosine(a,b):
    a,b=finite_arrays(a,b);den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a.ravel(),b.ravel())/den) if den else None


def update_diagnostics(before,after,prediction,ref_before,ref_after,ref_prediction,delta):
    before,after,prediction,ref_before,ref_after,ref_prediction=finite_arrays(before,after,prediction,ref_before,ref_after,ref_prediction)
    if type(delta) not in (float,int) or not np.isfinite(delta) or not -1<=delta<0:raise ValueError('Invalid Euler delta')
    error_before=before-ref_before;error_after=after-ref_after;injection=delta*(prediction-ref_prediction)
    residual=error_after-error_before-injection
    return dict(incoming_error_l2=float(np.linalg.norm(error_before)),outgoing_error_l2=float(np.linalg.norm(error_after)),
                prediction_injection_l2=float(np.linalg.norm(injection)),incoming_injection_cosine=cosine(error_before,injection),
                update_residual_l2=float(np.linalg.norm(residual)),update_residual_max=float(np.abs(residual).max()))


def schedule(steps):
    # These saved experiments use 8: every raw linspace value is exact binary FP32.
    if type(steps) is not int or steps!=8:raise ValueError('Only reviewed eight-step schedule supported')
    raw=np.arange(8,0,-1,dtype=np.float32)/np.float32(8)
    sigmas=np.append(np.float32(4)*raw/(np.float32(1)+np.float32(3)*raw),np.float32(0)).astype(np.float32)
    return np.diff(sigmas).astype(np.float64)


def read_tensor(path,digest,shape):
    path=Path(path)
    if any(type(x) is not int or x<1 for x in shape) or int(np.prod(shape,dtype=object))*4>128*1024*1024:raise ValueError('Unreviewed tensor shape/size')
    if sha(path)!=digest or path.stat().st_size!=int(np.prod(shape,dtype=object))*4:raise ValueError('Tensor SHA/size mismatch')
    value=np.fromfile(path,'<f4')
    if not np.isfinite(value).all():raise ValueError('Nonfinite tensor')
    return value


def analyze(runs):
    rows=[];common=None
    for run in map(Path,runs):
        audited=summarize_run(run) # fixed gates, runner/scripts, all tensor/PNG identities recomputed
        result=json.loads((run/'result.json').read_text());fixture=json.loads((run/'reference/fixture.json').read_text())
        identity=dict(reference_fixture=sha(run/'reference/fixture.json'),package=result['package_manifest_sha256'],
                      ids=canonical_sha256(fixture['ids']),config=fixture['config'],steps=fixture['steps'],initial=fixture['inputs']['initial']['sha256'])
        if common is None:common=identity
        if identity!=common or result['dit_precision']!='fp32' or result['device']!='vulkan':raise ValueError('Unmatched reference/input/package/precision/device identity')
        measured={v['tensor']:v for v in result['comparisons']}
        def native(name,entry):return read_tensor(run/'trace'/(name+'.f32'),measured[name]['sha256'],entry['shape'])
        def official(entry):
            if Path(entry['file']).name!=entry['file']:raise ValueError('Unsafe reference tensor path')
            return read_tensor(run/'reference'/entry['file'],entry['sha256'],entry['shape'])
        initial=fixture['inputs']['initial'];before=native('initial',initial);ref_before=official(initial);steps=[]
        for i,delta in enumerate(schedule(fixture['steps'])):
            entries=fixture['outputs'][i];after=native(f'step-{i}',entries['step']);ref_after=official(entries['step'])
            prediction=native(f'prediction-{i}',entries['prediction']);ref_prediction=official(entries['prediction'])
            stats=update_diagnostics(before,after,prediction,ref_before,ref_after,ref_prediction,float(delta))
            steps.append(dict(step=i,delta=float(delta),prediction=measured[f'prediction-{i}'],sample=measured[f'step-{i}'],**stats))
            before,ref_before=after,ref_after
        rows.append(dict(run=str(run.resolve()),result_sha256=sha(run/'result.json'),runner_sha256=result['runner_sha256'],
                         validator_sha256=result['validator_sha256'],conditioning_source=audited['conditioning_source'],
                         native_acceptance_eligible=audited['native_acceptance_eligible'],passed=audited['passed'],
                         failed_boundaries=audited['failed_boundaries'],text=measured['text'],final=measured['final'],decoded=measured['decoded'],png=audited['png'],steps=steps))
    if len(rows)<2:raise ValueError('Require at least two saved runs')
    return dict(schema_version=1,scope='Saved eight-step development traces only; no model execution or causal attribution',
                tool_sha256=sha(__file__),dependencies={p:sha(Path(__file__).parent/p) for p in ['diagnose_trajectory.py','collect_parity_evidence.py']},
                shared_identity=common,runs=rows,runner_identities_equal=len({r['runner_sha256'] for r in rows})==1,
                caveat='Different runner SHAs confound a text-only interpretation. Free-running prediction differences include latent drift, text perturbation and native numerical error.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new output file')
    result=analyze(a.run);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'runs':len(result['runs']),'runner_identities_equal':result['runner_identities_equal'],'output':str(a.output)}))
if __name__=='__main__':main()
