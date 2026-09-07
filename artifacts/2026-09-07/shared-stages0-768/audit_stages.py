"""Recompute all stage differences from authenticated complete executions."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

base=Path(__file__).resolve().parent
run=base/'native/diagnostic'
plan=json.loads((base/'plan.json').read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert sha(base/'plan.json')=='863167a7f536eb9fd2dc58c3f2af07d07f36e8798663842c383a7d6b9fbe81ce'
for name,item in plan['bindings'].items():
    path=Path(name);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
process=json.loads((base/'native/process.json').read_text())
worker=json.loads((base/'native/worker-result.json').read_text())
assert process['complete'] and process['return_code']==0 and process['plan_sha256']==sha(base/'plan.json')
assert len(worker)==1 and worker[0]['return_code']==0 and worker[0]['command']==plan['commands']['native'][0]
report=json.loads((run/'result.json').read_text())
fixture=json.loads((run/'oracle/fixture/fixture.json').read_text())
assert fixture['step']==0 and fixture['threads']==2 and fixture['valid_text_tokens']==15
assert fixture['native_acceptance_eligible'] is False and fixture['reference_fixture_sha256']==plan['reference_fixture_sha256']
assert report['native_acceptance_eligible'] is False and report['prediction']['return_code']==0
assert sha(run/'runner.snapshot')==plan['runner_sha256']==report['prediction']['runner_sha256']
for name,value in plan['package_binding'].items():assert fixture['package_binding'][name]==value,name
for entry in [*fixture['inputs'].values(),fixture['expected']]:
    assert sha(run/'oracle/fixture'/entry['file'])==entry['sha256']
prior=Path('/var/tmp/ernie-step0-768-v1/official/diagnostic/fixture')
prior_fixture=json.loads((prior/'fixture.json').read_text())
for name,entry in fixture['inputs'].items():assert entry['sha256']==prior_fixture['inputs'][name]['sha256'],name
reference_replay=fixture['expected']['sha256']==plan['criteria']['official_prediction_replay_sha256']
native_replay=sha(run/'native/actual.f32')==plan['criteria']['native_prediction_replay_sha256']
expected_names=[*(f'head-{i}' for i in range(8)),*(f'block-{i}' for i in range(36))]
assert list(fixture['stages'])==expected_names and len(report['stages'])==44
def values(path,shape,digest=None):
    assert path.stat().st_size==int(np.prod(shape))*4
    if digest:assert sha(path)==digest
    value=np.fromfile(path,'<f4').astype('f8').reshape(shape)
    assert np.isfinite(value).all()
    return value
def metric(actual,expected):
    delta=actual-expected
    norm=float(np.sqrt(np.sum(expected*expected)))
    return dict(max_abs_error=float(np.max(np.abs(delta))),
                nrmse=float(np.sqrt(np.sum(delta*delta))/max(norm,1e-15)),
                reference_max_abs=float(np.max(np.abs(expected))))
rows=[];count=0
for old,name in zip(report['stages'],expected_names):
    assert old['stage']==name
    entry=fixture['stages'][name]
    expected=values(run/'oracle/fixture'/entry['file'],entry['shape'],entry['sha256'])
    actual_path=run/'trace'/(name+'.f32')
    actual=values(actual_path,entry['shape'],old['actual_sha256'])
    error=metric(actual,expected)
    for key,value in error.items():assert math.isclose(value,old[key],rel_tol=1e-9,abs_tol=1e-12),(name,key)
    row=dict(stage=name,shape=entry['shape'],elements=actual.size,actual_sha256=sha(actual_path),expected_sha256=entry['sha256'],**error)
    if actual.size==2368*4096:
        for part,section in [('image_tokens',slice(0,2304)),('text_tokens',slice(2304,2368))]:
            result=metric(actual[:,section],expected[:,section])
            for key,value in result.items():assert math.isclose(value,old[part][key],rel_tol=1e-9,abs_tol=1e-12),(name,part,key)
            row[part]=result
    rows.append(row);count+=actual.size
expected=values(run/'oracle/fixture/expected.f32',[1,128,48,48])
actual=values(run/'native/actual.f32',[1,128,48,48])
prediction=metric(actual,expected)
gate=plan['criteria']['prediction_gate']
limit=gate['atol']+gate['global_rtol']*prediction['reference_max_abs']
prediction_pass=prediction['max_abs_error']<=limit and prediction['nrmse']<=gate['nrmse']
assert prediction_pass==report['prediction']['passed']
hidden=[row for row in rows if 'image_tokens' in row]
growth=[]
for left,right in zip(hidden,hidden[1:]):
    growth.append(dict(stage=right['stage'],previous=left['stage'],
                       nrmse_ratio=right['nrmse']/max(left['nrmse'],1e-30),
                       nrmse=right['nrmse'],max_abs_error=right['max_abs_error']))
record=dict(plan_sha256=sha(base/'plan.json'),verified_bindings=len(plan['bindings']),
 native_acceptance_eligible=False,complete_execution=True,stage_count=len(rows),
 finite_elements_per_side=count,official_prediction_exact_replay=reference_replay,
 native_prediction_exact_replay=native_replay,all_six_inputs_equal_previous_teacher=True,
 prediction=prediction,prediction_max_limit=limit,prediction_passed=bool(prediction_pass),
 stages=rows,largest_nrmse_growth=sorted(growth,key=lambda r:r['nrmse_ratio'],reverse=True)[:8],
 limitations='Stage growth includes propagated input error and local computation; it does not establish which operator introduced the discrepancy or additive causal contributions.',
 process=process)
with (base/'audit.json').open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')
print(json.dumps({key:record[key] for key in ('stage_count','finite_elements_per_side','official_prediction_exact_replay','native_prediction_exact_replay','prediction','largest_nrmse_growth')},indent=2))
raise SystemExit(0 if reference_replay and native_replay and prediction_pass else 1)
