"""Freeze final committed runtime and authenticate this run's exact inputs."""
from pathlib import Path
import hashlib, importlib.util, json, shutil, subprocess, tarfile

work = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
base = Path(__file__).resolve().parent
build = work/'build-dev'

def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'sha256': digest, 'bytes': path.stat().st_size}

def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

def git(*args):
    return subprocess.check_output(['git', *args], cwd=work)

assert not (base/'full-progress.json').exists(), 'Never refreeze an executed protocol'
assert not (base/'frozen-bin').exists(), 'Freeze is write-once'
head = git('rev-parse', 'HEAD').decode().strip()
spec = importlib.util.spec_from_file_location('inventory', work/'tools/source_inventory.py')
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)
files = inventory.source_files(work)
records = {}
for path in files:
    name = str(path.relative_to(work))
    record = identity(path)
    assert hashlib.sha256(git('show', head+':'+name)).hexdigest() == record['sha256'], name
    records[name] = record
assert not git('-C', 'third_party/ncnn', 'status', '--porcelain').strip(), 'Pinned ncnn must remain clean'
save(base/'source-bindings.json', {'head': head, 'files': records})
with tarfile.open(base/'source-snapshot.tar.gz', 'w:gz') as archive:
    for path in files:
        archive.add(path, arcname=str(path.relative_to(work)), recursive=False)
(base/'frozen-bin').mkdir()
binary = base/'frozen-bin/ernie-image'
shutil.copy2(build/'ernie-image', binary)
binary.chmod(0o555)
for name in ['CMakeCache.txt', 'compile_commands.json']:
    shutil.copy2(build/name, base/name)
commands = json.loads((base/'compile_commands.json').read_text())
derived = {}
for item in commands:
    path = Path(item['file'])
    if path.is_relative_to(build) and path.name in {'allocator.cpp', 'command.cpp', 'net.cpp', 'sdpa_vulkan.cpp'}:
        derived[str(path.relative_to(build))] = {**identity(path), 'compile_command': item['command']}
assert len(derived) == 4, derived
save(base/'derived-compiled-units.json', {'files': derived})
plan = json.loads((base/'full-plan.json').read_text())
plan['execution_source_head'] = head
plan['execution_source_state'] = 'All inventoried sources verified against committed Git objects before binary freeze'
plan['current_ncnn'] = git('-C', 'third_party/ncnn', 'rev-parse', 'HEAD').decode().strip()
plan['status'] = 'frozen before execution; results recorded separately'
sdk = Path(plan['validation']['runtime'])
for path in [binary, base/'source-bindings.json', base/'source-snapshot.tar.gz', base/'derived-compiled-units.json',
             base/'CMakeCache.txt', base/'compile_commands.json', base/'run_full.py', base/'compare_full.py',
             base/'historical-fp32-anchor.json', base/'freeze.py',
             sdk/'share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json',
             sdk/'lib/libVkLayer_khronos_validation.so', sdk/'bin/vulkaninfo']:
    plan['bindings'][str(path)] = identity(path)
save(base/'full-plan.json', plan)
for index, (path, record) in enumerate(plan['bindings'].items(), 1):
    assert identity(path) == record, path
    if index % 25 == 0:
        print('Verified', index, 'bindings', flush=True)
save(base/'preflight.json', {'all_passed': True, 'source_files': len(records), 'source_head': head,
    'bindings_verified': len(plan['bindings']), 'plan_sha256': identity(base/'full-plan.json')['sha256'],
    'binary': identity(binary), 'derived_compiled_units': len(derived)})
print((base/'preflight.json').read_text(), flush=True)
