import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan')
WORK = ROOT / '.worktrees/surpass-reference'
OLD = ROOT / 'outputs/runtime-rectangles-v2/portrait'
sys.path.insert(0, str(WORK / 'tools'))
from source_inventory import source_files

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=WORK)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK, text=True).strip()
assert head == '990e8ef07591c2f7564d538462d51b8961263655'
source = BASE / 'source'
source.mkdir()
for path in source_files(WORK):
    target = source / path.relative_to(WORK)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    target.chmod(0o444)
(source / 'models').symlink_to(ROOT / 'models', target_is_directory=True)
binary = BASE / 'ernie-image.snapshot'
shutil.copyfile(WORK / 'build-dev/ernie-image', binary)
binary.chmod(0o555)
assert sha(binary) == '92559ea493adabfb1a0b2fbf1d69e21ccb79d7c9f2dbd37cb25898d7654da1c0'
shutil.copyfile(WORK / 'build-dev/CMakeCache.txt', BASE / 'CMakeCache.snapshot.txt')
ncnn = ROOT / 'third_party/ncnn'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ncnn)
identity = {'source_head': head, 'source_files': len(source_files(source)),
            'binary_sha256': sha(binary), 'binary_bytes': binary.stat().st_size,
            'ncnn_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ncnn, text=True).strip(),
            'ncnn_archive_sha256': sha(WORK / 'build-dev/ncnn/src/libncnn.a'),
            'compact_reader': False, 'default_mapped_loading': False,
            'scope': 'Current Linux Vulkan build with Windows path fixes; no inference math changes; not the older memory grid or compact-reader binary'}
save(BASE / 'build-identity.json', identity)
common_files = [*source_files(source), binary, BASE / 'CMakeCache.snapshot.txt', BASE / 'build-identity.json', Path(__file__)]
shared = ROOT / 'models/turbo-shared-v2'
common_files.append(shared / 'manifest.json')
old_plan = json.loads((OLD / 'plan.json').read_text())
common_files.extend(Path(name) for name in old_plan['bindings'] if '/models/' in name and '/source/' not in name)
# The loaders authenticate actual official weight bytes against these manifests.
# Bind the manifests and configuration too, so neither side can silently change.
official = ROOT / 'models/official'
for prefix in ('text-', 'dit-', 'vae-'):
    common_files.extend(official.glob(prefix + '*.manifest.json'))
common_files.extend(p for p in official.glob('*config.json') if p.is_file())
common_files.extend(p for p in (ROOT / 'models/tokenizer').iterdir() if p.is_file())
common = {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(set(common_files))}
gates = old_plan['gates']
commands = []
for size in (768, 1024):
    case = BASE / str(size)
    case.mkdir()
    (case / 'source').symlink_to(source, target_is_directory=True)
    for name in ('model', 'temporary', 'cache'):
        (case / name).mkdir()
    initial = case / 'initial.f32'
    np.random.Generator(np.random.PCG64(20260905)).standard_normal((1, 128, size // 16, size // 16), dtype=np.float32).astype('<f4').tofile(initial)
    request = {'package': str(ROOT / 'models/portable-turbo1024-s32-v1'),
               'source_weights_package': str(ROOT / 'models/turbo1024-s64-portable'),
               'prompt': 'A red apple on a wooden table, soft daylight, realistic photo.',
               'steps': 8, 'runtime_size': [size, size], 'initial': str(initial),
               'source_head': head, 'reference_device': 'cuda',
               'initial_generation': 'NumPy PCG64 seed20260905; saved little-endian FP32 bytes shared by native/official; no cross-framework seed equivalence claim'}
    save(case / 'request.json', request)
    shutil.copyfile(OLD / 'worker.py', case / 'worker.py')
    shutil.copyfile('/var/tmp/ernie-runtime512-native-v1/run.py', case / 'run.py')
    supervisor = (OLD / 'supervisor.py').read_text().replace("unit = 'ernie-rect-portrait-v2-' + phase", "unit = 'ernie-square-" + str(size) + "-v1-' + phase")
    (case / 'supervisor.py').write_text(supervisor)
    comparator = (OLD / 'compare.py').read_text().replace("native=Path('/var/tmp/ernie-runtime-rectangles-v1/portrait')", 'native=base')
    comparator = comparator.replace('Installed native text and 8-step runtime384x512 trajectory', f'Native text and 8-step runtime{size}x{size} trajectory with explicit RAM weights/stdio/cache0')
    comparator += '\nraise SystemExit(0 if result["passed"] else 1)\n'
    (case / 'compare.py').write_text(comparator)
    bindings = {**common}
    for name in ('initial.f32', 'request.json', 'worker.py', 'run.py', 'supervisor.py', 'compare.py'):
        path = case / name
        path.chmod(0o444)
        bindings[str(path)] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    native = ['bwrap', '--die-with-parent', '--unshare-net', '--ro-bind', '/', '/',
              '--bind', str(case), str(case), '--ro-bind', str(shared), str(case / 'model'),
              '--tmpfs', str(ROOT), '--dev-bind', '/dev', '/dev', '--proc', '/proc',
              '--chdir', str(case), '--setenv', 'TMPDIR', str(case / 'temporary'),
              '--setenv', 'XDG_CACHE_HOME', str(case / 'cache'), '--', str(binary),
              '--model', str(case / 'model'), '--prompt', request['prompt'],
              '--output', str(case / 'native/native.png'), '--trace-dir', str(case / 'native/trace'),
              '--report-json', str(case / 'native/report.json'),
              '--width', str(size), '--height', str(size), '--latent', str(initial),
              '--steps', '8', '--threads', '2', '--device', 'vulkan', '--precision', 'fp32',
              '--text-down-vector', '--vae-device', 'cpu', '--vae-convolution', 'direct',
              '--dit-weights', 'host', '--model-loading', 'stdio', '--dit-cache-mib', '0']
    official_command = ['bwrap', '--die-with-parent', '--unshare-net', '--ro-bind', '/', '/',
                        '--bind', str(case), str(case), '--dev-bind', '/dev', '/dev', '--proc', '/proc',
                        '--chdir', str(case), '--setenv', 'TMPDIR', str(case / 'temporary'),
                        '--setenv', 'XDG_CACHE_HOME', str(case / 'cache'), '--',
                        str(WORK / '.venv/bin/python'), str(case / 'worker.py')]
    plan = {'scope': f'Development runtime{size}x{size} native and official FP32 full trajectory; identical saved latent and source32, 64 DiT text slots; not peer performance, auto-pressure, OOM recovery or broad quality acceptance',
            'commands': {'native': [native], 'official': [official_command]},
            'resource_limits': old_plan['resource_limits'], 'gates': gates,
            'model_manifest_sha256': sha(shared / 'manifest.json'),
            'binary_sha256': sha(binary), 'source_head': head,
            'source_isolated_native': True, 'network_disabled_both': True,
            'bindings': bindings}
    save(case / 'plan.json', plan)
    (case / 'plan.json').chmod(0o444)
    for phase in ('native', 'official', 'compare'):
        argv = [str(WORK / '.venv/bin/python'), str(case / ('compare.py' if phase == 'compare' else 'supervisor.py'))]
        if phase != 'compare': argv.append(phase)
        commands.append({'case': str(size), 'phase': phase, 'argv': argv})
    print(json.dumps({'case': size, 'plan_sha256': sha(case / 'plan.json'), 'bindings': len(bindings), 'initial_sha256': sha(initial)}), flush=True)
save(BASE / 'commands.json', commands)
shutil.copyfile(ROOT / 'outputs/runtime-rectangles-v1/run_batch.py', BASE / 'run_batch.py')
save(BASE / 'batch-identity.json', {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in [BASE / 'commands.json', BASE / 'run_batch.py', *(BASE / str(s) / 'plan.json' for s in (768, 1024))]})
