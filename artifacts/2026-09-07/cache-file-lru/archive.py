"""Verify frozen identities and preserve small records from the completed run."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

root = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
out = root / 'outputs/cache-file-lru-v1'
base = Path('/var/tmp/ernie-cache-file-lru512-v1')
dest = root / 'artifacts/2026-09-07/cache-file-lru'

def identity(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}

plan = json.loads((base / 'plan.json').read_text())
for name, record in plan['bindings'].items():
    assert identity(Path(name)) == record, name
comparison = json.loads((base / 'comparison.json').read_text())
assert comparison['passed']
assert comparison['plan_sha256'] == identity(base / 'plan.json')['sha256']
process = json.loads((base / 'native/process.json').read_text())
assert process['complete'] and process['return_code'] == 0
assert process['observed_native_pids']
assert all(not Path('/proc', str(pid)).exists() for pid in process['observed_native_pids'])
events = dict((key, int(value)) for key, value in
              (line.split() for line in process['memory_events'].splitlines()))
assert events['oom'] == events['oom_kill'] == 0
assert process['observed_limits']['memory.max'] == '17179869184'
assert process['observed_limits']['memory.swap.max'] == '0'

checks = json.loads((out / 'checks.json').read_text())
assert all(row['returncode'] == 0 for row in checks)
test_runs = []
for row in checks:
    if row['command'][0] != 'ctest':
        continue
    log = Path(row['log']).read_text()
    count = re.search(r'100% tests passed, 0 tests failed out of (\d+)', log)
    assert count and '***Not Run' not in log and '***Skipped' not in log
    test_runs.append({'log': Path(row['log']).name, 'passed': int(count[1]),
                      'command': row['command']})
assert [row['passed'] for row in test_runs] == [7, 3]
ncnn = root.parent.parent / 'third_party/ncnn'
revision = subprocess.check_output(['git', '-C', str(ncnn), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == '6a1bf000f363714839a36793addc8c879d3d899e'
assert not subprocess.check_output(['git', '-C', str(ncnn), 'status', '--porcelain'], text=True)
verification = {
    'frozen_bindings_verified': len(plan['bindings']),
    'plan_sha256': identity(base / 'plan.json')['sha256'],
    'comparison_sha256': identity(base / 'comparison.json')['sha256'],
    'native_pids_finished': process['observed_native_pids'],
    'ncnn_revision': revision, 'ncnn_clean': True,
    'test_runs': test_runs, 'final_policy_checks': 10,
    'memory_events': events,
    'scope': 'One traced development regression; no paired performance, activation spilling or general OOM recovery claim.',
}
(out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')

dest.mkdir(parents=True, exist_ok=False)
records = {}

def retain(source, name):
    assert source.is_file() and name not in records
    target = dest / name
    shutil.copy2(source, target)
    records[name] = {'source': str(source), **identity(target)}

for name in ('plan.json', 'build-identity.json', 'run.py', 'supervisor.py', 'comparison.json'):
    retain(base / name, name)
for name in ('process.json', 'worker-result.json', 'command-0.log', 'runner.log'):
    retain(base / 'native' / name, 'native-' + name)
for name in ('model-loading.txt', 'weight-cache.txt', 'weight-placement.txt', 'shape.txt'):
    retain(base / 'native/trace' / name, name)
for name in ('prepare.py', 'compare.py', 'check.py', 'archive.py', 'checks.json', 'verification.json', 'policy-basis.json', 'sample-replay.json'):
    retain(out / name, name)
for path in sorted(out.glob('*.log')):
    retain(path, path.name)
(dest / 'inventory.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps({'destination': str(dest), 'records': len(records),
                  'bytes': sum(r['bytes'] for r in records.values()),
                  'frozen_bindings_verified': len(plan['bindings']),
                  'final_policy_checks': 10, 'cache': comparison['weight_cache']}, indent=2))
