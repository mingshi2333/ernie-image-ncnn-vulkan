"""Recompute every saved boundary in FP64 chunks independently of torch metrics."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

start = time.monotonic()
ref = BASE / 'official/validation/reference'
native = BASE / 'native/validation'
fixture = json.loads((ref / 'fixture.json').read_text())
worker = json.loads((native / 'result.json').read_text())
plan = json.loads((BASE / 'plan.json').read_text())
assert fixture['complete'] is True and len(fixture['outputs']) == 8
assert worker['return_code'] == 0
assert worker['conditioning_source'] == 'native_text_encoder'
assert worker['native_acceptance_eligible'] is True
assert worker['text_down_reduction'] == 'vector' and worker['dit_precision'] == 'fp32'
assert worker['reference_fixture_sha256'] == sha(ref / 'fixture.json')
assert worker['runner_sha256'] == plan['runner_sha256'] == sha(native / 'ernie-image.snapshot')
assert worker['package_manifest_sha256'] == plan['model_manifest_sha256']
gates = json.loads((native / 'gates.json').read_text())
assert all(gates[k] == v for k, v in plan['gates'].items())
ids = [int(v) for v in (native / 'trace/ids.txt').read_text().split()]
assert ids == fixture['ids']
entries = [(k, v, True) for k, v in fixture['inputs'].items()]
for i, stage in enumerate(fixture['outputs']):
    entries.extend((f'{k}-{i}', v, False) for k, v in stage.items())
entries.extend((k, v, False) for k, v in fixture['final'].items())
assert len(entries) == len(worker['comparisons']) == 25
assert [k for k, _, _ in entries] == [v['tensor'] for v in worker['comparisons']]
rows = []
for (name, item, conditioning), original in zip(entries, worker['comparisons']):
    a_path = ref / item['file']
    b_path = native / 'trace' / (name + '.f32')
    count = math.prod(item['shape'])
    assert a_path.stat().st_size == b_path.stat().st_size == count * 4
    assert sha(a_path) == item['sha256'] and sha(b_path) == original['sha256']
    a = np.memmap(a_path, '<f4', mode='r')
    b = np.memmap(b_path, '<f4', mode='r')
    squared = reference_squared = maximum = reference_maximum = 0.
    for offset in range(0, count, 1 << 20):
        expected = a[offset:offset + (1 << 20)].astype('f8')
        actual = b[offset:offset + (1 << 20)].astype('f8')
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        difference = actual - expected
        squared += float(np.dot(difference, difference))
        reference_squared += float(np.dot(expected, expected))
        maximum = max(maximum, float(abs(difference).max()))
        reference_maximum = max(reference_maximum, float(abs(expected).max()))
    nrmse = math.sqrt(squared / max(reference_squared, 1e-30))
    gate = gates['conditioning' if conditioning else 'fp32']
    limit = gate['atol'] + gate['global_rtol'] * reference_maximum
    passed = nrmse <= gate['nrmse'] and maximum <= limit
    if name == 'initial':
        passed = sha(a_path) == sha(b_path) == sha(ref / 'initial.f32')
    assert math.isclose(nrmse, original['nrmse'], rel_tol=1e-10, abs_tol=1e-14)
    assert maximum == original['max_abs_error'] and bool(passed) == original['passed']
    rows.append({'tensor': name, 'shape': item['shape'], 'elements': count,
                 'all_values_finite': True, 'nrmse': nrmse, 'max_abs_error': maximum,
                 'max_abs_limit': limit, 'reference_sha256': sha(a_path), 'native_sha256': sha(b_path), 'passed': bool(passed)})
    del a, b
expected = np.asarray(Image.open(ref / 'reference.png').convert('RGB'))
actual = np.asarray(Image.open(native / 'native.png').convert('RGB'))
assert expected.shape == actual.shape == (768, 1376, 3)
delta = abs(actual.astype('f8') - expected.astype('f8'))
png = {'mae': float(delta.mean()), 'max_abs': float(delta.max()), 'shape': list(actual.shape),
       'sha256': sha(native / 'native.png'), 'reference_sha256': sha(ref / 'reference.png')}
png['passed'] = png['mae'] <= gates['fp32']['pixel_mae'] and png['max_abs'] <= gates['fp32']['pixel_max']
decoded = np.memmap(native / 'trace/decoded.f32', '<f4', mode='r', shape=(3, 768, 1376))
quantized = (np.clip(decoded / 2 + .5, 0, 1).transpose(1, 2, 0) * 255).round().astype('uint8')
png_exact = np.array_equal(quantized, actual)
assert png_exact == worker['native_png_quantization_exact']
assert all(png[k] == worker['png'][k] for k in ('mae', 'max_abs', 'shape', 'sha256', 'passed'))
passed = png_exact and png['passed'] and all(row['passed'] for row in rows)
assert bool(passed) == worker['passed']
report = {'complete_native_generation': True, 'passed_original_gates': bool(passed),
          'scope': 'One real prompt, 25 native CPU text layers, 8 Vulkan FP32 denoising steps, CPU direct VAE; PE off, CFG=1, saved initial noise; root recomputation',
          'reviewer_source_sha256': sha(__file__), 'plan_sha256': sha(BASE / 'plan.json'),
          'validator_result_sha256': sha(native / 'result.json'),
          'reference_fixture_sha256': sha(ref / 'fixture.json'), 'token_ids_exact': True, 'token_count': len(ids),
          'tensor_denominator': len(rows), 'elements_compared': sum(row['elements'] for row in rows),
          'tensors_passed': sum(row['passed'] for row in rows), 'comparisons': rows,
          'png': png, 'native_png_quantization_exact': bool(png_exact),
          'root_review_wall_seconds': time.monotonic() - start,
          'formal_quality_or_peer_acceptance': False}
(BASE / 'actual-review.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'comparisons'}), flush=True)
raise SystemExit(0 if passed else 1)
