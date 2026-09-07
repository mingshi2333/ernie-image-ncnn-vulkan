"""Check the frozen compact-reader full-pipeline regression without performance claims."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

base = Path(__file__).resolve().parent
plan = json.loads((base / 'plan.json').read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


for name, record in plan['bindings'].items():
    path = Path(name)
    assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name
process = json.loads((base / 'native/process.json').read_text())
assert process['complete'] and process['return_code'] == 0, process
worker = json.loads((base / 'native/worker-result.json').read_text())
assert len(worker) == 1 and worker[0]['return_code'] == 0
assert worker[0]['command'] == plan['commands']['native'][0]
report = json.loads((base / 'native/generation.json').read_text())
assert report['schema_version'] == 1 and report['status'] == 'success'
assert report['trace_enabled'] is True and report['allocation_instrumentation'] is False
assert report['shape'] == [512, 512] and report['shape_order'] == 'WH'
assert report['model'] == plan['expected_model']
assert report['request'] == plan['expected_request']
assert report['prompt'] == plan['prompt'] and report['token_ids'] == plan['token_ids']
assert report['vulkan_gpu_index'] == 0 and report['pe']['enabled'] is False
assert report['model_loading_requested'] == 'stdio'
assert report['placement_requests'] == {'gpu': 0, 'ram': 304, 'budget_unavailable': 0}
assert all(value == 0 for value in report['weight_cache'].values())
assert [(p['current'], p['total']) for p in report['progress'] if p['stage'] == 'denoise'] == [(i, 8) for i in range(1, 9)]

before, after = base / 'baseline/trace', base / 'native/trace'
names = {p.name for p in before.glob('*.f32')}
assert names == {p.name for p in after.glob('*.f32')} and len(names) == 25
rows = []
for name in sorted(names):
    a, b = np.fromfile(before / name, '<f4'), np.fromfile(after / name, '<f4')
    assert a.shape == b.shape
    rows.append({'file': name, 'elements': int(a.size),
                 'finite': bool(np.isfinite(a).all() and np.isfinite(b).all()),
                 'baseline_sha256': sha(before / name), 'candidate_sha256': sha(after / name),
                 'bitwise_equal': sha(before / name) == sha(after / name),
                 'maximum_abs_difference': float(np.max(np.abs(a.astype('float64') - b.astype('float64'))))})
for name in ('initial.f32', 'text.f32', 'ids.txt', 'prompt.txt'):
    assert sha(before / name) == sha(after / name), name
a = np.array(Image.open(base / 'baseline/native.png'))
b = np.array(Image.open(base / 'native/native.png'))
assert a.shape == b.shape == (512, 512, 3)
png = {'baseline_sha256': sha(base / 'baseline/native.png'),
       'candidate_sha256': sha(base / 'native/native.png'),
       'pixels_equal': bool(np.array_equal(a, b)),
       'maximum_channel_difference': int(np.max(np.abs(a.astype('int16') - b.astype('int16'))))}
png['bitwise_equal'] = png['baseline_sha256'] == png['candidate_sha256']
assert png['baseline_sha256'] == 'bf87214d21461bfda96529830b28a5b7630ef0dcf307c0172f1666fc64044e28'
events = dict((key, int(value)) for key, value in (line.split() for line in process['memory_events'].splitlines()))
log = (base / 'native/command-0.log').read_text()
result = {'scope': plan['scope'], 'plan_sha256': sha(base / 'plan.json'),
          'float_tensors': len(rows), 'total_elements': sum(row['elements'] for row in rows),
          'tensors': rows, 'png': png, 'memory_events': events,
          'placement_requests': report['placement_requests'],
          'allocator_fallback_logged': 'weight allocator fallback to device memory' in log,
          'process': process, 'formal_speed_or_memory_result': False}
result['passed'] = (all(row['finite'] and row['bitwise_equal'] for row in rows)
                    and png['bitwise_equal'] and not result['allocator_fallback_logged']
                    and events['oom'] == 0 and events['oom_kill'] == 0)
(base / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('tensors', 'process')}, indent=2))
raise SystemExit(0 if result['passed'] else 1)
