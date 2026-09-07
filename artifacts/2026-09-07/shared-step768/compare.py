import hashlib
import json
from pathlib import Path
import sys
import numpy as np

base = Path(__file__).resolve().parent
phase = sys.argv[1]
assert phase in ('native', 'official')
plan = json.loads((base / 'plan.json').read_text())
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
for name, expected in plan['bindings'].items():
    path = Path(name)
    assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
process = json.loads((base / phase / 'process.json').read_text())
worker = json.loads((base / phase / 'worker-result.json').read_text())
assert process['complete'] and process['return_code'] == 0 and process['plan_sha256'] == sha(base / 'plan.json')
assert len(worker) == 1 and worker[0]['command'] == plan['commands'][phase][0] and worker[0]['return_code'] == 0
runtime = [json.loads(line) for line in (base / phase / 'command-0.log').read_text().splitlines() if line.startswith('{')][-1]
assert all(runtime[k] == v for k, v in dict(backend='vulkan', precision='fp32', policy='stream',
    blocks=36, threads=2, package_mode=True, text_bucket=32, valid_text_tokens=15,
    host_weights=True, shared_pipeline_cache=True, dit_heads=True).items())
expected_item = plan['fixtures'][phase]['expected']
expected_path = base / (phase + '-fixture') / expected_item['file']
actual_path = base / phase / 'actual.f32'
assert sha(expected_path) == expected_item['sha256']
actual = np.fromfile(actual_path, '<f4').astype('f8')
expected = np.fromfile(expected_path, '<f4').astype('f8')
assert actual.shape == expected.shape == (294912,) and np.isfinite(actual).all() and np.isfinite(expected).all()
delta = actual - expected
maximum = float(np.abs(delta).max())
nrmse = float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-30))
exact = sha(actual_path) == sha(expected_path)
passed = exact
limit = None
if phase == 'official':
    replay = json.loads((base / 'native-comparison.json').read_text())
    assert replay['exact_bytes'] and replay['criterion_passed'] and replay['plan_sha256'] == sha(base / 'plan.json')
    gate = plan['criteria']['official']
    limit = gate['atol'] + gate['global_rtol'] * float(np.abs(expected).max())
    passed = maximum <= limit and nrmse <= gate['nrmse']
result = dict(phase=phase, native_acceptance_eligible=False, plan_sha256=sha(base / 'plan.json'),
    source_head=plan['source_head'], runner_sha256=sha(base / 'runner.snapshot'),
    verified_bindings=len(plan['bindings']), complete_execution=True, finite_elements=actual.size,
    actual_sha256=sha(actual_path), expected_sha256=sha(expected_path), exact_bytes=exact,
    nrmse=nrmse, max_abs=maximum, max_abs_limit=limit, criterion_passed=bool(passed),
    runtime=runtime, process=process)
with (base / (phase + '-comparison.json')).open('x') as stream:
    stream.write(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:result[k] for k in ('phase','complete_execution','finite_elements','exact_bytes','nrmse','max_abs','max_abs_limit','criterion_passed')}))
raise SystemExit(0 if passed else 1)
