"""One maximum-range real-weight block comparison; no generator limit change."""
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

BASE = Path(__file__).resolve().parent
PLAN = json.loads((BASE / 'plan.json').read_text())
PHASE = sys.argv[1]
sys.path.insert(0, str(BASE / 'source/tools'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(actual_path, expected_path, elements):
    assert Path(actual_path).stat().st_size == Path(expected_path).stat().st_size == elements * 4
    a = np.memmap(actual_path, dtype='<f4', mode='r')
    b = np.memmap(expected_path, dtype='<f4', mode='r')
    error = reference = maximum = reference_maximum = 0.
    for start in range(0, elements, 262144):
        x = a[start:start + 262144].astype(np.float64)
        y = b[start:start + 262144].astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite block output')
        delta = x - y
        error += float(np.dot(delta, delta))
        reference += float(np.dot(y, y))
        maximum = max(maximum, float(np.max(np.abs(delta))))
        reference_maximum = max(reference_maximum, float(np.max(np.abs(y))))
    gate = PLAN['gates']['fp32']
    nrmse = math.sqrt(error / max(reference, 1e-30))
    return {'elements': elements, 'all_finite': True, 'nrmse': nrmse,
            'max_abs_error': maximum, 'reference_max_abs': reference_maximum,
            'passed_historical_gate': nrmse <= gate['nrmse'] and
                maximum <= gate['atol'] + gate['rtol'] * reference_maximum}


if PHASE == 'official':
    import torch
    from export_dit_block import load_block, make_inputs, save_tensor, ErnieImageSharedAdaLNBlock
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    block, weights = load_block(Path(PLAN['official_weights']), 0)
    assert weights['sha256'] == PLAN['official_weights_sha256']
    assert weights['revision'] == PLAN['official_revision']
    implementation = Path(inspect.getfile(ErnieImageSharedAdaLNBlock)).resolve()
    assert sha(implementation) == PLAN['official_implementation_sha256']
    config = PLAN['input_config']
    inputs, frequencies = make_inputs(config['height'], config['width'],
                                     config['text_tokens'], config['valid_text'], config['seed'])
    reference = BASE / 'official/reference'
    reference.mkdir()
    records = {f'in{i}': save_tensor(reference / f'in{i}.f32', value)
               for i, value in enumerate(inputs)}
    print('Saved ten actual official inputs for 10240 tokens', flush=True)
    start = time.monotonic()
    expected = block(inputs[0].transpose(0, 1), frequencies,
                     list(inputs[1:7]), attention_mask=inputs[-1][None, None]).transpose(0, 1)
    seconds = time.monotonic() - start
    entry = save_tensor(reference / 'expected.f32', expected)
    finite = np.isfinite(expected.detach().numpy()).all()
    if not finite:
        raise ValueError('Nonfinite official result')
    fixture = {'complete': True, 'tokens': 10240, 'shape': [1, 10240, 4096],
               'inputs': records, 'expected': entry, 'gates': PLAN['gates'],
               'input_config': config, 'official_weights_sha256': weights['sha256'],
               'official_revision': weights['revision'],
               'official_implementation_path': str(implementation),
               'official_implementation_sha256': sha(implementation),
               'torch_runtime': torch.__version__, 'torch_threads': torch.get_num_threads(),
               'device': 'cpu', 'seconds': seconds,
               'scope': 'One actual official block, synthetic hidden/AdaLN and full-length text positions; no heads, denoising trajectory or image'}
    (reference / 'fixture.json').write_text(json.dumps(fixture, indent=2) + '\n')
    print(json.dumps({'complete': True, 'elements': expected.numel(),
                      'all_finite': bool(finite), 'seconds': seconds}), flush=True)
elif PHASE == 'native':
    reference = BASE / 'official/reference'
    fixture_path = reference / 'fixture.json'
    fixture_sha = sha(fixture_path)
    fixture = json.loads(fixture_path.read_text())
    assert fixture['complete'] and fixture['tokens'] == 10240
    assert fixture['input_config'] == PLAN['input_config'] and fixture['gates'] == PLAN['gates']
    assert fixture['official_weights_sha256'] == PLAN['official_weights_sha256']
    for entry in [*fixture['inputs'].values(), fixture['expected']]:
        assert Path(entry['file']).name == entry['file']
        path = reference / entry['file']
        assert sha(path) == entry['sha256']
        assert path.stat().st_size == math.prod(entry['shape']) * 4
    output = BASE / 'native/out0.f32'
    command = [str(BASE / 'ernie-block-sequence-runner.snapshot'),
               '--model', str(BASE / 'model'), '--fixture', str(reference),
               '--tokens', '10240', '--backend', 'vulkan', '--precision', 'fp32',
               '--policy', 'stream', '--large-shape-probe', '--output', str(output)]
    with (BASE / 'native/native.log').open('xb') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    report = {'command': command, 'return_code': result.returncode,
              'reference_fixture_sha256': fixture_sha, 'passed_historical_gate': False}
    if result.returncode == 0:
        report.update(compare(output, reference / fixture['expected']['file'], 10240 * 4096))
        report['native_sha256'] = sha(output)
        report['reference_sha256'] = fixture['expected']['sha256']
        assert sha(fixture_path) == fixture_sha
        for entry in [*fixture['inputs'].values(), fixture['expected']]:
            assert sha(reference / entry['file']) == entry['sha256']
    (BASE / 'native/comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report['passed_historical_gate'] else 1)
else:
    raise ValueError('Unknown phase')
