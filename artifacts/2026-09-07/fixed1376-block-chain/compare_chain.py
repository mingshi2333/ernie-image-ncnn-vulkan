#!/usr/bin/env python3
"""Recompute all connected block outputs in bounded FP64 chunks."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
REFERENCE = BASE / 'official-chain/reference'
NATIVE = BASE / 'native-chain'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


fixture = json.loads((REFERENCE / 'fixture.json').read_text())
plan = json.loads((BASE / 'plan.json').read_text())
assert fixture['gates'] == plan['chain_gates']
assert fixture['blocks'] == list(range(36))
assert [s['block'] for s in fixture['reference_stages']] == list(range(36))
assert {p.name for p in (NATIVE / 'trace').iterdir()} == {f'block-{i}.f32' for i in range(36)}
rows = []
for stage in fixture['reference_stages']:
    index = stage['block'];entry = stage['output']
    assert entry['shape'] == [1, 4192, 4096]
    ref = REFERENCE / entry['file'];actual = NATIVE / 'trace' / f'block-{index}.f32'
    assert sha(ref) == entry['sha256']
    x = np.memmap(actual, '<f4', mode='r');y = np.memmap(ref, '<f4', mode='r')
    assert x.size == y.size == 17170432
    squares = [];references = [];maximum = magnitude = 0.0
    for i in range(0, x.size, 1048576):
        a = x[i:i+1048576].astype(np.float64);b = y[i:i+1048576].astype(np.float64)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        delta = a-b
        squares.append(float(np.sum(delta*delta)));references.append(float(np.sum(b*b)))
        maximum = max(maximum, float(np.abs(delta).max()));magnitude = max(magnitude, float(np.abs(b).max()))
    nrmse = math.sqrt(math.fsum(squares)) / max(math.sqrt(math.fsum(references)), 1e-30)
    limits = plan['chain_gates']['fp32'];limit = limits['atol'] + limits['rtol']*magnitude
    rows.append({'block': index, 'elements': int(x.size), 'finite': True, 'nrmse': nrmse,
                 'max_abs_error': maximum, 'max_abs_limit': limit, 'nrmse_limit': limits['nrmse'],
                 'passed': maximum <= limit and nrmse <= limits['nrmse'],
                 'reference_sha256': entry['sha256'], 'actual_sha256': sha(actual)})
    del x, y
assert sha(NATIVE/'validation/actual.f32') == rows[-1]['actual_sha256']
single = json.loads((BASE/'native-single/root-review.json').read_text())
assert rows[0]['actual_sha256'] == single['actual_sha256']
assert rows[0]['reference_sha256'] == json.loads((BASE/'official-single/reference/fixture.json').read_text())['expected']['sha256']
original = json.loads((NATIVE/'validation/result.json').read_text())
assert abs(rows[-1]['nrmse']-original['nrmse']) < 1e-14 and rows[-1]['max_abs_error'] == original['max_abs_error']
passed = sum(r['passed'] for r in rows)
result = {'status': 'passed_connected_blocks' if passed == 36 else 'quality_gate_failed',
          'passed_blocks': passed, 'total_blocks': 36, 'elements_compared': sum(r['elements'] for r in rows),
          'rows': rows, 'first_block_reproduces_single_reference_and_native': True,
          'final_native_matches_last_trace': True, 'comparator_sha256': sha(Path(__file__)),
          'reference_fixture_sha256': sha(REFERENCE/'fixture.json'),
          'formal_speed_eligible': False, 'formal_memory_eligible': False,
          'scope': 'One free-running connected36 block chain with real weights and synthetic hidden/AdaLN; no heads, text encoder, Euler, VAE or full1376 image'}
(BASE/'chain-root-review.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: result[k] for k in ['status','passed_blocks','total_blocks','elements_compared']}), flush=True)
raise SystemExit(0 if passed == 36 else 1)
