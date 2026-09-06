"""Verify frozen inputs, run one original validator phase, and preserve failures."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
phase, expected_plan = sys.argv[1:]
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
assert sha(BASE / 'plan.json') == expected_plan
plan = json.loads((BASE / 'plan.json').read_text())
def verify():
    for name, item in plan['bindings'].items():
        path = Path(name)
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Frozen binding changed: ' + name)
    runtime = json.loads((BASE / 'runtime-identity.json').read_text())
    for name, version in runtime['distribution_versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Installed reference version differs: ' + name)
start = time.monotonic()
verify()
if phase == 'native':
    fixture = json.loads((BASE / 'official/validation/reference/fixture.json').read_text())
    assert fixture['complete'] and len(fixture['outputs']) == 8
    assert fixture['config'] == plan['config']
print(json.dumps({'phase': phase, 'frozen_bindings_verified': len(plan['bindings']), 'launching_model': True}), flush=True)
result = subprocess.run(plan['commands'][phase])
verify()
gates = json.loads((BASE / phase / 'validation/gates.json').read_text())
assert all(gates[k] == v for k, v in plan['gates'].items())
if phase == 'official' and result.returncode == 0:
    fixture = json.loads((BASE / 'official/validation/reference/fixture.json').read_text())
    assert fixture['reference_environment']['torch'] == json.loads((BASE / 'runtime-identity.json').read_text())['versions']['torch']
record = {'phase': phase, 'return_code': result.returncode, 'plan_sha256': expected_plan,
          'bindings_verified_before_and_after': len(plan['bindings']), 'wall_seconds': time.monotonic() - start,
          'status': 'completed' if result.returncode == 0 else 'validation_failed_preserved'}
(BASE / phase / 'worker-result.json').write_text(json.dumps(record, indent=2) + '\n')
raise SystemExit(result.returncode)
