"""Recompute completed runtime-target parity and its unchanged fixed-package output."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


plan = json.loads((BASE / 'plan.json').read_text())
assert sha(BASE / 'plan.json') == '50aa7b633c3e17b28e7be9d17a099b1044782ecaa305d8ff64710b95d1709106'
run = BASE / 'native/validation'
result = json.loads((run / 'result.json').read_text())
old_result_path = Path(plan['historical_native_result'])
old = json.loads(old_result_path.read_text())
fixture_path = run / 'reference/fixture.json'
fixture = json.loads(fixture_path.read_text())
assert sha(fixture_path) == result['reference_fixture_sha256'] == '930d1593ea6e4246e03edd990fbdf3a805de1b378ea2a5bbc8b15695a762f128'
assert result['return_code'] == 0
assert result['runner_sha256'] == plan['runner_sha256'] == sha(run / 'ernie-image.snapshot')
assert result['text_down_reduction'] == old['text_down_reduction']
assert result['package_manifest_sha256'] == plan['shared_manifest_sha256']
assert fixture['config'] == plan['target_config']
entries = dict(fixture['inputs'])
for index, step in enumerate(fixture['outputs']):
    entries.update({name + '-' + str(index): item for name, item in step.items()})
entries.update(fixture['final'])
assert len(entries) == len(result['comparisons']) == len(old['comparisons']) == 25
recorded = {item['tensor']: item for item in result['comparisons']}
historical = {item['tensor']: item for item in old['comparisons']}
assert entries.keys() == recorded.keys() == historical.keys()
gates = json.loads((run / 'gates.json').read_text())
assert gates == json.loads((old_result_path.parent / 'gates.json').read_text())
comparisons = []
for name, item in entries.items():
    reference_path = run / 'reference' / item['file']
    native_path = run / 'trace' / (name + '.f32')
    fixed_path = old_result_path.parent / 'trace' / (name + '.f32')
    reference_sha = sha(reference_path)
    native_sha = sha(native_path)
    assert reference_sha == item['sha256']
    assert native_sha == recorded[name]['sha256']
    assert sha(fixed_path) == historical[name]['sha256']
    count = math.prod(item['shape'])
    assert reference_path.stat().st_size == native_path.stat().st_size == count * 4
    reference = np.memmap(reference_path, dtype='<f4', mode='r')
    actual = np.memmap(native_path, dtype='<f4', mode='r')
    squared_error = squared_reference = maximum = reference_maximum = 0.
    for start in range(0, count, 262144):
        a = actual[start:start + 262144].astype(np.float64)
        b = reference[start:start + 262144].astype(np.float64)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        delta = a - b
        squared_error += float(np.dot(delta, delta))
        squared_reference += float(np.dot(b, b))
        maximum = max(maximum, float(np.max(np.abs(delta))))
        reference_maximum = max(reference_maximum, float(np.max(np.abs(b))))
    nrmse = math.sqrt(squared_error / max(squared_reference, 1e-30))
    gate = gates['conditioning' if name in fixture['inputs'] else 'fp32']
    passed = nrmse <= gate['nrmse'] and maximum <= gate['atol'] + gate['global_rtol'] * reference_maximum
    if name == 'initial':
        passed = native_sha == reference_sha
    assert passed == recorded[name]['passed']
    assert math.isclose(nrmse, recorded[name]['nrmse'], rel_tol=1e-10, abs_tol=1e-15)
    assert maximum == recorded[name]['max_abs_error']
    comparisons.append({'tensor': name, 'elements': count, 'sha256': native_sha,
                        'nrmse': nrmse, 'max_abs_error': maximum, 'historical_gate_passed': passed,
                        'bitwise_equal_to_fixed_package': native_sha == historical[name]['sha256']})
    del reference, actual
native_png = np.array(Image.open(run / 'native.png').convert('RGB'))
reference_png = np.array(Image.open(run / 'reference/reference.png').convert('RGB'))
fixed_png = np.array(Image.open(old_result_path.parent / 'native.png').convert('RGB'))
assert native_png.shape == reference_png.shape == fixed_png.shape == (768, 1376, 3)
error = np.abs(native_png.astype(np.int16) - reference_png.astype(np.int16))
decoded = np.fromfile(run / 'trace/decoded.f32', dtype='<f4').reshape(3, 768, 1376)
quantized = (np.clip(decoded / 2 + .5, 0, 1).transpose(1, 2, 0) * 255).round().astype(np.uint8)
assert np.array_equal(native_png, quantized)
assert float(error.mean()) == result['png']['mae'] and int(error.max()) == result['png']['max_abs']
assert sha(run / 'native.png') == result['png']['sha256']
values, counts = np.unique(error, return_counts=True)
review = {'scope': plan['scope'], 'native_execution_return_code': result['return_code'],
          'source_files': plan['source_file_count'], 'plan_bindings': len(plan['bindings']),
          'all_compared_tensors_finite': True, 'tensor_elements': sum(c['elements'] for c in comparisons),
          'comparisons': comparisons, 'numerical_gate_passed': result['passed'],
          'tensor_gates_passed': sum(c['historical_gate_passed'] for c in comparisons),
          'all_25_tensors_equal_fixed_package': all(c['bitwise_equal_to_fixed_package'] for c in comparisons),
          'png_equal_fixed_package': bool(np.array_equal(native_png, fixed_png)),
          'png_file_equal_fixed_package': sha(run / 'native.png') == sha(old_result_path.parent / 'native.png'),
          'native_png_quantization_exact': True, 'png_mae': float(error.mean()),
          'png_max_abs': int(error.max()), 'png_sha256': sha(run / 'native.png'),
          'png_error_histogram': {str(v): int(c) for v, c in zip(values, counts)}}
assert review['tensor_elements'] == 32098304
(BASE / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
print(json.dumps({key: value for key, value in review.items() if key != 'comparisons'}))
