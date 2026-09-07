"""Compare cached RAM weights with the frozen original image and preceding RAM run."""
import hashlib, json, re
from pathlib import Path
import numpy as np
from PIL import Image

base = Path('/var/tmp/ernie-cache-file-lru512-v1')
plan = json.loads((base/'plan.json').read_text())
baseline = Path(plan['baseline_run'])
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
for name, record in plan['bindings'].items():
    p = Path(name)
    assert p.stat().st_size == record['bytes'] and sha(p) == record['sha256'], name
process = json.loads((base/'native/process.json').read_text())
assert process['complete'], process
assert len(process['observed_model_mappings']) >= plan['expected_mapping_minimum']
assert all(row['permissions'] == ['r--p'] for row in process['observed_model_mappings'].values())
before, after = baseline/'native/trace', base/'native/trace'
names = {p.name for p in before.glob('*.f32')}
assert names == {p.name for p in after.glob('*.f32')} and len(names) == 25
rows = []
for name in sorted(names):
    a, b = np.fromfile(before/name, '<f4'), np.fromfile(after/name, '<f4')
    assert a.shape == b.shape
    row = {'file': name, 'elements': int(a.size), 'finite': bool(np.isfinite(a).all() and np.isfinite(b).all()),
           'baseline_sha256': sha(before/name), 'cached_ram_sha256': sha(after/name)}
    row['bitwise_equal'] = row['baseline_sha256'] == row['cached_ram_sha256']
    row['maximum_abs_difference'] = float(np.max(np.abs(a.astype('float64') - b.astype('float64'))))
    rows.append(row)
for name in ('initial.f32', 'text.f32', 'ids.txt', 'prompt.txt'):
    assert sha(before/name) == sha(after/name), name
image_a, image_b = baseline/'native/native.png', base/'native/native.png'
assert sha(image_a) == 'bf87214d21461bfda96529830b28a5b7630ef0dcf307c0172f1666fc64044e28'
a, b = np.array(Image.open(image_a)), np.array(Image.open(image_b))
assert a.shape == b.shape == (512, 512, 3)
png = {'baseline_sha256': sha(image_a), 'cached_ram_sha256': sha(image_b),
       'pixels_equal': bool(np.array_equal(a, b)), 'maximum_channel_difference': int(np.max(np.abs(a.astype('int16')-b.astype('int16'))))}
png['bitwise_equal'] = png['baseline_sha256'] == png['cached_ram_sha256']
cache = dict((key, int(value)) for key, value in (line.split('=', 1) for line in (after/'weight-cache.txt').read_text().splitlines()))
assert cache['hits'] >= plan['expected_cache']['minimum_hits']
assert cache['hits'] + cache['loads'] == plan['expected_cache']['total_block_calls'] == 288
assert 0 < cache['peak_charged_bytes'] <= plan['expected_cache']['budget_bytes']
assert cache['unavailable_queries'] == 0
placements = (after/'weight-placement.txt').read_text().splitlines()
assert len(placements) == 16 + cache['loads']
assert all(' requested=host ' in line for line in placements)
assert all('reason=cache ' in line or 'reason=budget ' in line for line in placements)
assert all('available_bytes=unavailable' not in line for line in placements)
available = [int(re.search(r'available_bytes=(\d+)', line).group(1)) for line in placements]
log = (base/'native/command-0.log').read_text()
assert 'Image model loading requested: mapped' in log
assert (after/'model-loading.txt').read_text().startswith('requested=mapped\n')
assert 'weight allocator fallback to device memory' not in log
assert 'GPU=0 RAM=' + str(len(placements)) + ' budget unavailable=0' in log
prior = Path(plan['previous_ram_run'])
for name in sorted(names):
    assert sha(prior/'native/trace'/name) == sha(before/name), name
prior_process = json.loads((prior/'native/process.json').read_text())
result = {'scope': plan['scope'], 'plan_sha256': sha(base/'plan.json'), 'float_tensors': len(rows),
          'total_elements': sum(r['elements'] for r in rows), 'tensors': rows, 'png': png,
          'weight_cache': cache, 'observed_model_mapping_files': len(process['observed_model_mappings']),
          'model_loading_requested': 'mapped',
          'weight_requests': {'host': len(placements), 'device': 0, 'unavailable': 0,
                              'minimum_available_bytes': min(available), 'maximum_available_bytes': max(available),
                              'allocator_fallback_logged': False},
          'previous_combined_wall_seconds': prior_process['wall_seconds'],
          'previous_cache': dict((k,int(v)) for k,v in (line.split('=',1) for line in (prior/'native/trace/weight-cache.txt').read_text().splitlines())),
          'sampled_native_memory': process['peak_sampled_native_memory'],
          'cgroup_memory_stat_peaks': process['peak_cgroup_memory_stat'],
          'process': {k: process[k] for k in ('wall_seconds', 'peak_memory_current', 'minimum_host_available',
                                             'peak_gpu_whole_device_mib', 'memory_events', 'return_code')},
          'passed': all(r['bitwise_equal'] and r['finite'] for r in rows) and png['bitwise_equal']}
goal = plan['retention_goal']
result['retention_goal'] = {'more_hits': cache['hits'] > goal['more_hits_than'],
                            'fewer_evictions': cache['evictions'] < goal['fewer_evictions_than'],
                            'prior_hits': goal['more_hits_than'], 'prior_evictions': goal['fewer_evictions_than']}
result['numerical_identity_passed'] = result['passed']
result['passed'] = result['passed'] and result['retention_goal']['more_hits'] and result['retention_goal']['fewer_evictions']
(base/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'tensors'}, indent=2))
raise SystemExit(0 if result['passed'] else 1)
