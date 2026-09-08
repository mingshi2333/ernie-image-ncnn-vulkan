import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

base = Path(__file__).resolve().parent
work = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
cases = []
for tokens in (288, 4160):
    for precision in ('fp32', 'fp16', 'bf16'):
        cases.append((f's{tokens}-{precision}', work / f'models/converted/dit-block-00-s{tokens}/runtime', precision))
for block, version in ((31, 3), (33, 4), (35, 4)):
    cases.append((f'block{block}-s288-bf16', work / f'models/dit-s288-v{version}/block-{block}/runtime', 'bf16'))
rows = []
for name, model, precision in cases:
    pair = {}
    for version in ('baseline', 'candidate'):
        out = base / 'blocks' / name / version
        out.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(work / 'tools/validate_dit_block.py'), '--model', str(model),
            '--runner', str(base / f'{version}-bin/ernie-block-runner'), '--output', str(out),
            '--backend', 'vulkan', '--precision', precision, '--device-io', '--host-weights', '--repeat', '1']
        with (out.parent / f'{version}-validation.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record = json.loads((out / 'result.json').read_text())
        record['validator_exit'] = result.returncode
        pair[version] = record
        print(name, version, 'passed', record['passed'], 'nrmse', record.get('nrmse'), flush=True)
        if record.get('return_code') != 0 or 'failure' in record:
            raise RuntimeError(f'{name}/{version} did not execute: {record}')
    a = np.fromfile(base / 'blocks' / name / 'baseline/actual.f32', '<f4')
    b = np.fromfile(base / 'blocks' / name / 'candidate/actual.f32', '<f4')
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    rows.append({'case': name, 'baseline': pair['baseline'], 'candidate': pair['candidate'],
        'elements': a.size, 'bitwise_equal': pair['baseline']['actual_sha256'] == pair['candidate']['actual_sha256'],
        'maximum_abs_difference': float(np.max(np.abs(a.astype('float64') - b.astype('float64'))))})
    (base / 'block-results.json').write_text(json.dumps(rows, indent=2) + '\n')
print('Completed', len(rows), 'matched real-weight cases; bitwise equal', sum(r['bitwise_equal'] for r in rows), flush=True)
