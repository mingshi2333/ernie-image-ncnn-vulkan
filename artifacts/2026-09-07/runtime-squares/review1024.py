import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

base = Path(__file__).resolve().parent
case = base / '1024'
sys.path.insert(0, str(base / 'source/tools'))
from pipeline_reference import full_reference_contract
from pipeline_package import select_shared_instance

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

plan = json.loads((case / 'plan.json').read_text())
request = json.loads((case / 'request.json').read_text())
for name, expected in plan['bindings'].items():
    path = Path(name)
    assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
processes = {phase: json.loads((case / phase / 'process.json').read_text()) for phase in ('native', 'official')}
for phase, result in processes.items():
    assert result['complete'] and result['return_code'] == 0
    assert result['plan_sha256'] == sha(case / 'plan.json')
    worker = json.loads((case / phase / 'worker-result.json').read_text())
    assert len(worker) == 1 and worker[0]['return_code'] == 0 and worker[0]['command'] == plan['commands'][phase][0]
ref = case / 'official/reference'
trace = case / 'native/trace'
fixture = json.loads((ref / 'fixture.json').read_text())
report = json.loads((case / 'native/report.json').read_text())
shared_path = next(Path(p) for p, identity in plan['bindings'].items()
                   if identity['sha256'] == plan['model_manifest_sha256'])
shared = json.loads(shared_path.read_text())
selected, runtime = select_shared_instance(shared['instances'], 1024, 1024, len(fixture['ids']))
assert full_reference_contract(fixture, selected['config'], request['prompt'], 8) == 25
assert selected['source_manifest_sha256'] == fixture['runtime_source']['source_manifest_sha256']
assert sha(Path(request['package']) / 'manifest.json') == selected['source_manifest_sha256']
assert sha(ref / 'initial-input.f32') == sha(case / 'initial.f32') == sha(trace / 'initial.f32')
assert fixture['source_sha256'] == sha(base / 'source/tools/validate_pipeline.py')
assert report['status'] == 'success' and report['shape'] == [1024, 1024]
assert report['token_ids'] == fixture['ids'] == [int(v) for v in (trace / 'ids.txt').read_text().split()]
assert report['placement_requests'] == {'gpu': 0, 'ram': 304, 'budget_unavailable': 0}
assert report['trace_enabled'] and not report['allocation_instrumentation']
assert {k: report['request'][k] for k in ('device', 'precision', 'vae_device', 'vae_convolution', 'text_device', 'text_down_vector', 'threads', 'steps', 'dit_weights', 'dit_cache_mib', 'model_loading')} == {
    'device': 'vulkan', 'precision': 'fp32', 'vae_device': 'cpu', 'vae_convolution': 'direct',
    'text_device': 'cpu', 'text_down_vector': True, 'threads': 2, 'steps': 8,
    'dit_weights': 'host', 'dit_cache_mib': 0, 'model_loading': 'stdio'}
assert not report['pe']['enabled']
assert [p['current'] for p in report['progress'] if p['stage'] == 'denoise'] == list(range(1, 9))
entries = [(name, item, 'conditioning') for name, item in fixture['inputs'].items()]
entries += [(f'{name}-{index}', item, 'fp32') for index, group in enumerate(fixture['outputs']) for name, item in group.items()]
entries += [(name, item, 'fp32') for name, item in fixture['final'].items()]
rows = []
for name, item, gate_name in entries:
    expected_path, actual_path = ref / item['file'], trace / (name + '.f32')
    assert sha(expected_path) == item['sha256']
    expected = np.fromfile(expected_path, '<f4').astype('float64')
    actual = np.fromfile(actual_path, '<f4').astype('float64')
    assert len(expected) == len(actual) == int(np.prod(item['shape']))
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    difference = actual - expected
    nrmse = float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-15))
    maximum = float(np.max(np.abs(difference)))
    gate = plan['gates'][gate_name]
    limit = gate['atol'] + gate['global_rtol'] * float(np.max(np.abs(expected)))
    passed = np.array_equal(actual, expected) if name == 'initial' else nrmse <= gate['nrmse'] and maximum <= limit
    rows.append({'name': name, 'elements': int(actual.size), 'finite': True, 'nrmse': nrmse,
                 'max_abs': maximum, 'max_abs_limit': limit, 'passed': bool(passed)})
original = json.loads((case / 'comparison.json').read_text())
assert [(r['name'], r['passed']) for r in rows] == [(r['name'], r['passed']) for r in original['comparisons']]
native_png, official_png = case / 'native/native.png', ref / 'reference.png'
assert sha(official_png) == fixture['reference_png_sha256']
a = np.asarray(Image.open(native_png).convert('RGB'), dtype='int16')
b = np.asarray(Image.open(official_png).convert('RGB'), dtype='int16')
assert a.shape == b.shape == (1024, 1024, 3)
delta = np.abs(a - b)
assert float(delta.mean()) == original['png']['mae'] and int(delta.max()) == original['png']['max_abs']
review = {'scope': 'Independent root re-read of complete1024 sources, inputs, execution, all25 finite tensors and PNG; no model rerun',
          'source_head': plan['source_head'], 'plan_sha256': sha(case / 'plan.json'),
          'reference_fixture_sha256': sha(ref / 'fixture.json'), 'verified_bindings': len(plan['bindings']),
          'complete_execution_both': True, 'rows': rows, 'finite_elements': sum(r['elements'] for r in rows),
          'passed_tensors': sum(r['passed'] for r in rows), 'png_mae': float(delta.mean()), 'png_max_abs': int(delta.max()),
          'png_passed': original['png']['passed'], 'all_quality_gates_passed': False,
          'native_png_sha256': sha(native_png), 'official_png_sha256': sha(official_png),
          'processes': processes}
assert len(rows) == 25 and review['passed_tensors'] == 24 and review['finite_elements'] == 31720448
with (base / 'review1024.json').open('x') as stream:
    stream.write(json.dumps(review, indent=2) + '\n')
print(json.dumps({k: review[k] for k in ('reference_fixture_sha256', 'verified_bindings', 'complete_execution_both', 'finite_elements', 'passed_tensors', 'png_mae', 'png_max_abs', 'all_quality_gates_passed')}, indent=2))
