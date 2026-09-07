"""Freeze the already-built reader candidate; leave GPU execution unstarted."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

root = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
base = Path('/var/tmp/ernie-compact-reader512-v1')
prior = Path('/var/tmp/ernie-cache-file-lru512-v1')
baseline = Path('/var/tmp/ernie-runtime512-native-v1/native')
evidence = root / 'artifacts/2026-09-07/compact-model-reader'
candidate = root / 'outputs/compact-model-reader-v1/candidate/vulkan/ernie-image'


def identity(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


assert identity(candidate)['sha256'] == 'b15ccce8e92962b041df6581926dd7c06f1f75b65b1b2cbb949f927be71ca05a'
source = json.loads((evidence / 'source-inventory.json').read_text())['files']
archive = subprocess.check_output(['git', 'archive', '8158dc153d7bb4a20d2f5cf8d3d2ca764a8db524', '--', *source], cwd=root)
base.mkdir()
for name in ('source', 'model', 'temporary', 'cache', 'baseline/trace'):
    (base / name).mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    for name, expected in source.items():
        data = tar.extractfile(name).read()
        assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], name
        path = base / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
shutil.copy2(candidate, base / 'ernie-image.snapshot')
shutil.copy2(prior / 'initial.f32', base / 'initial.f32')
assert identity(base / 'initial.f32')['sha256'] == '1c89a002f18b3162b5a43100dc6f50bdd247dd0af8d1944ec87705153819d453'
shutil.copy2(prior / 'run.py', base / 'run.py')
supervisor = (prior / 'supervisor.py').read_text().replace('ernie-cache-file-lru512-v1-', 'ernie-compact-reader512-v1-')
insertion = '''# A waiting/failed/partial grid cannot authorize a second GPU workload.
grid = Path('/var/tmp/ernie-memory-grid512-v1')
assert hashlib.file_digest((grid / 'grid.json').open('rb'), 'sha256').hexdigest() == '04d0e4e7454436853e4031e1eee55d367cfb2a2edd33038c71eb7319081d8a1b'
progress = json.loads((grid / 'progress.json').read_text())
assert progress['status'] == 'complete' and len(progress['finished_trials']) == 16, 'Existing memory comparison has not completed'
assert all(trial['status'] == 'ok' for trial in progress['finished_trials'])
old_master = Path('/proc') / str(progress['master_pid'])
if old_master.exists():
    assert str(grid / 'master.py').encode() not in (old_master / 'cmdline').read_bytes()
grid_result = json.loads((grid / 'results.json').read_text())
assert grid_result['status'] == 'complete' and grid_result['completed_runs'] == 16 and grid_result['failed_runs'] == 0
'''
assert "phase = sys.argv[1]\n" in supervisor
supervisor = supervisor.replace("phase = sys.argv[1]\n", "phase = sys.argv[1]\n" + insertion)
(base / 'supervisor.py').write_text(supervisor)
for name in ('prepare.py', 'compare.py'):
    shutil.copy2(root / 'outputs/compact-reader512-v1' / name, base / name)
for path in (baseline / 'trace').iterdir():
    if path.is_file(): shutil.copy2(path, base / 'baseline/trace' / path.name)
for name in ('native.png', 'process.json'):
    shutil.copy2(baseline / name, base / 'baseline' / name)
for name in ('source-inventory.json', 'candidate-identity.json', 'source-checks.json', 'model-reader.patch'):
    shutil.copy2(evidence / name, base / name)
shutil.copy2(root / 'build-dev/compact-model-reader/modelbin.cpp', base / 'derived-modelbin.cpp')
assert identity(base / 'derived-modelbin.cpp')['sha256'] == '90adcd77b0a3c2cf49142d65f6a8dc74dadc26ef7460c162c2b21721234f7a69'
old = json.loads((prior / 'plan.json').read_text())
command = [arg.replace(str(prior), str(base)) for arg in old['commands']['native'][0]]
for flag, value in (('--dit-weights', 'host'), ('--gpu-reserve-mib', '512'),
                    ('--dit-cache-mib', '0'), ('--model-loading', 'stdio')):
    command[command.index(flag) + 1] = value
command += ['--report-json', str(base / 'native/generation.json')]
request = json.loads(Path('/var/tmp/ernie-memory-grid512-v1/00-warmup-mapped-off/native/benchmark/generation.json').read_text())
request['request']['model_loading'] = 'stdio'
manifest = root / 'models/turbo-shared-v2/manifest.json'
assert identity(manifest)['sha256'] == '21b6bc8deae17418c22257b52e928ee048a372342a2517bcb954276257c21d1a'
bindings = {str(path): identity(path) for path in sorted(base.rglob('*')) if path.is_file()}
bindings[str(manifest)] = identity(manifest)
plan = {'scope': 'Full compact-reader candidate regression: saved512/native Vector text/8 Vulkan FP32 steps/CPU direct VAE, stdio and explicit host weights, cacheOFF. TraceON; no speed or whole-process memory claim.',
        'commands': {'native': [command]}, 'resource_limits': old['resource_limits'], 'bindings': bindings,
        'expected_model': request['model'], 'expected_request': request['request'],
        'prompt': request['prompt'], 'token_ids': request['token_ids'],
        'source_commit': '8158dc153d7bb4a20d2f5cf8d3d2ca764a8db524',
        'prerequisite': 'All sixteen existing memory-grid runs must finish successfully and the master must exit. This preparation does not start a model job.'}
(base / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
for name in ('source', 'baseline'):
    for path in (base / name).rglob('*'):
        if path.is_file(): path.chmod(0o444)
for path in base.iterdir():
    if path.is_file(): path.chmod(0o555 if path.name == 'ernie-image.snapshot' else 0o444)
print(json.dumps({'base': str(base), 'plan': identity(base / 'plan.json'), 'bindings': len(bindings), 'gpu_job_started': False}, indent=2))
