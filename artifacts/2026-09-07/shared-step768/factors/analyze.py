import hashlib
import json
from pathlib import Path
import numpy as np

base = Path(__file__).resolve().parent
plan = json.loads((base / 'plan.json').read_text())
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
for name, item in plan['bindings'].items():
    path = Path(name)
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
def values(path):
    value = np.fromfile(path, '<f4').astype('f8')
    assert value.shape == (294912,) and np.isfinite(value).all()
    return value
references = {name: values(path) for name, path in plan['references'].items()}
def metrics(actual, reference):
    delta = actual - reference
    return dict(nrmse=float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-30)),
                max_abs=float(np.abs(delta).max()))
results = []
for phase in ('latent', 'text'):
    process = json.loads((base / phase / 'process.json').read_text())
    worker = json.loads((base / phase / 'worker-result.json').read_text())
    assert process['complete'] and process['return_code'] == 0 and process['plan_sha256'] == sha(base / 'plan.json')
    assert len(worker) == 1 and worker[0]['return_code'] == 0 and worker[0]['command'] == plan['commands'][phase][0]
    runtime = [json.loads(line) for line in (base / phase / 'command-0.log').read_text().splitlines() if line.startswith('{')][-1]
    assert all(runtime[k] == v for k, v in dict(backend='vulkan', precision='fp32', policy='stream',
        blocks=36, threads=2, package_mode=True, text_bucket=32, valid_text_tokens=15,
        host_weights=True, shared_pipeline_cache=True, dit_heads=True).items())
    actual_path = base / phase / 'actual.f32'
    actual = values(actual_path)
    results.append(dict(phase=phase, changed_input=plan['fixtures'][phase]['changed_input'],
        finite_elements=actual.size, actual_sha256=sha(actual_path),
        difference_from_native=metrics(actual, references['native']),
        descriptive_distance_from_official=metrics(actual, references['official']),
        complete_execution=True, process=process, runtime=runtime))
record = dict(scope=plan['scope'], native_acceptance_eligible=False, official_gate_applied=False,
    plan_sha256=sha(base / 'plan.json'), verified_bindings=len(plan['bindings']),
    saved_native_distance_from_official=metrics(references['native'], references['official']),
    factors=results)
with (base / 'analysis.json').open('x') as stream:
    stream.write(json.dumps(record, indent=2) + '\n')
print(json.dumps({'baseline':record['saved_native_distance_from_official'],
    'factors':[{k:r[k] for k in ('phase','changed_input','descriptive_distance_from_official')} for r in results]},indent=2))
