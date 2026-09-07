"""Keep numerical agreement separate from the predeclared cache-reuse expectation.

This additional analysis was written during the combined run after observing
cgroup file-cache pressure. It preserves the frozen checker and its result.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

root = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
out = root / 'outputs/runtime-model-loading-v1'
base = Path('/var/tmp/ernie-runtime-mapped-cache512-v1')
plan = json.loads((base / 'plan.json').read_text())

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

for name, item in plan['bindings'].items():
    path = Path(name)
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
with (out / 'frozen-comparison.log').open('w') as stream:
    frozen = subprocess.run([sys.executable, str(out / 'compare.py')], stdout=stream, stderr=subprocess.STDOUT)
process = json.loads((base / 'native/process.json').read_text())
assert process['complete'] and process['return_code'] == 0
before = Path(plan['baseline_run']) / 'native/trace'
after = base / 'native/trace'
names = sorted(p.name for p in before.glob('*.f32'))
assert set(names) == {p.name for p in after.glob('*.f32')} and len(names) == 25
rows = []
for name in names:
    a, b = np.fromfile(before / name, '<f4'), np.fromfile(after / name, '<f4')
    rows.append({'name': name, 'elements': int(a.size), 'same_shape': a.shape == b.shape,
                 'finite': bool(np.isfinite(a).all() and np.isfinite(b).all()),
                 'byte_equal': sha(before / name) == sha(after / name)})
cache = dict((k, int(v)) for k, v in (line.split('=', 1) for line in (after / 'weight-cache.txt').read_text().splitlines()))
png = base / 'native/native.png'
baseline_png = Path(plan['baseline_run']) / 'native/native.png'
checks = {
    'finite_exact_25_tensors': all(r['finite'] and r['same_shape'] and r['byte_equal'] for r in rows),
    'exact_baseline_png': sha(png) == sha(baseline_png),
    'all_288_block_calls_accounted': cache['hits'] + cache['loads'] == plan['expected_cache']['total_block_calls'],
    'cache_charge_within_budget': 0 < cache['peak_charged_bytes'] <= plan['expected_cache']['budget_bytes'],
    'minimum_cache_hits_predeclared': cache['hits'] >= plan['expected_cache']['minimum_hits'],
    'observed_readonly_mappings': bool(process['observed_model_mappings']) and
        all(r['permissions'] == ['r--p'] for r in process['observed_model_mappings'].values()),
}
result = {'scope': 'Additional review, with the original minimum-hit expectation unchanged; not formal performance.',
          'frozen_checker_returncode': frozen.returncode, 'checks': checks, 'weight_cache': cache,
          'tensors': rows, 'total_elements': sum(r['elements'] for r in rows), 'png_sha256': sha(png),
          'plan_sha256': sha(base / 'plan.json'), 'review_sha256': sha(Path(__file__)),
          'observed_mapping_files': len(process['observed_model_mappings']),
          'process': {key: process[key] for key in ('wall_seconds', 'peak_memory_current',
              'minimum_host_available', 'peak_gpu_whole_device_mib', 'memory_events', 'return_code')},
          'all_checks_pass': all(checks.values())}
(out / 'review.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'tensors'}, indent=2))
