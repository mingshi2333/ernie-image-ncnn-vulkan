"""Independently verify the frozen first-step executions and raw predictions."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

base=Path(__file__).resolve().parent
phase=sys.argv[1]
assert phase in ('native','official')
plan=json.loads((base/'plan.json').read_text())
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
def metric(actual,expected):
    delta=actual-expected
    return dict(nrmse=float(np.linalg.norm(delta)/max(np.linalg.norm(expected),1e-30)),
                max_abs=float(np.abs(delta).max()))
for name,item in plan['bindings'].items():
    path=Path(name)
    assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
process=json.loads((base/phase/'process.json').read_text())
worker=json.loads((base/phase/'worker-result.json').read_text())
assert process['complete'] and process['return_code']==0 and process['plan_sha256']==sha(base/'plan.json')
assert len(worker)==1 and worker[0]['command']==plan['commands'][phase][0] and worker[0]['return_code']==0
official_command=plan['commands']['official'][0]
reference=Path(official_command[official_command.index('--reference')+1])
oracle=json.loads((reference/'fixture.json').read_text())
official_path=reference/oracle['outputs'][0]['prediction']['file']
expected_path=base/'native-fixture/expected.f32' if phase=='native' else official_path
directory=base/phase if phase=='native' else base/phase/'diagnostic/native'
runtime_log=directory/('command-0.log' if phase=='native' else 'runner.log')
runtime=[json.loads(line) for line in runtime_log.read_text().splitlines() if line.startswith('{')][-1]
assert all(runtime[k]==v for k,v in dict(backend='vulkan',precision='fp32',policy='stream',blocks=36,
 threads=2,package_mode=True,text_bucket=32,valid_text_tokens=15,host_weights=True,
 shared_pipeline_cache=True,dit_heads=True).items())
actual_path=directory/'actual.f32'
actual=np.fromfile(actual_path,'<f4').astype('f8')
expected=np.fromfile(expected_path,'<f4').astype('f8')
official=np.fromfile(official_path,'<f4').astype('f8')
assert all(a.shape==(294912,) and np.isfinite(a).all() for a in (actual,expected,official))
error=metric(actual,expected)
exact=sha(actual_path)==sha(expected_path)
passed=exact;maximum_limit=None
if phase=='official':
    replay=json.loads((base/'native-comparison.json').read_text())
    assert replay['exact_bytes'] and replay['criterion_passed'] and replay['plan_sha256']==sha(base/'plan.json')
    gate=plan['criteria']['official']
    maximum_limit=gate['atol']+gate['global_rtol']*float(np.abs(expected).max())
    passed=error['nrmse']<=gate['nrmse'] and error['max_abs']<=maximum_limit
    diagnostic=base/phase/'diagnostic'
    fixture=json.loads((diagnostic/'fixture/fixture.json').read_text())
    result=json.loads((directory/'result.json').read_text())
    assert fixture['step']==0 and fixture['timestep']==1000 and fixture['config']==oracle['config']
    assert fixture['native_acceptance_eligible'] is False and fixture['threads']==2 and fixture['host_weights_requested']
    assert fixture['reference_fixture_sha256']==plan['reference_fixture_sha256']
    for entry in [*fixture['inputs'].values(),fixture['expected']]:
        assert sha(diagnostic/'fixture'/entry['file'])==entry['sha256']
    assert sha(diagnostic/'runner.snapshot')==plan['runner_sha256']==result['runner_sha256']
    assert result['passed']==passed and result['return_code']==0 and result['actual_sha256']==sha(actual_path)
    assert abs(result['nrmse']-error['nrmse'])<1e-12 and abs(result['max_abs_error']-error['max_abs'])<1e-12
    assert abs(result['max_abs_limit']-maximum_limit)<1e-12
report=dict(phase=phase,step=0,native_acceptance_eligible=False,plan_sha256=sha(base/'plan.json'),
 runner_sha256=plan['runner_sha256'],verified_bindings=len(plan['bindings']),complete_execution=True,
 finite_elements=actual.size,actual_sha256=sha(actual_path),expected_sha256=sha(expected_path),
 exact_bytes=exact,**error,max_abs_limit=maximum_limit,criterion_passed=bool(passed),
 distance_to_official=metric(actual,official),runtime=runtime,process=process)
with (base/(phase+'-comparison.json')).open('x') as stream:
    stream.write(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('phase','complete_execution','exact_bytes','nrmse','max_abs','max_abs_limit','criterion_passed','distance_to_official')}))
raise SystemExit(0 if passed else 1)
