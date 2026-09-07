import hashlib
import json
from pathlib import Path
import subprocess
import time

base = Path(__file__).resolve().parent
assert not Path('/proc/3130217').exists(), 'Original batch must be terminal'
assert not (base / '1024/native').exists(), 'Never overwrite/restart a model run'
for phase in ('native', 'official'):
    assert json.loads((base / '768' / phase / 'process.json').read_text())['complete']
comparison = json.loads((base / '768/comparison.json').read_text())
assert comparison['passed'] is False and comparison['passed_tensors'] == 18
for name, expected in json.loads((base / 'batch-identity.json').read_text()).items():
    path = Path(name)
    assert path.stat().st_size == expected['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256']
results = []
for step in json.loads((base / 'commands.json').read_text()):
    if step['case'] != '1024':
        continue
    print(json.dumps({'starting': step['case'] + ' ' + step['phase']}), flush=True)
    start = time.monotonic()
    with (base / (step['case'] + '-' + step['phase'] + '.log')).open('xb') as log:
        process = subprocess.run(step['argv'], stdout=log, stderr=subprocess.STDOUT)
    results.append({**step, 'return_code': process.returncode, 'wall_seconds': time.monotonic() - start})
    (base / 'continuation1024-results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results[-1]), flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)
