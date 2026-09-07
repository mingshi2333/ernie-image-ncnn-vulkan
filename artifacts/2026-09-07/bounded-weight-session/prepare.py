"""Freeze the bounded prepared-weight cache on the saved 512 input."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
prior = Path('/var/tmp/ernie-weight-placement512-v1')
base = Path('/var/tmp/ernie-weight-cache512-v1')
sys.path.insert(0, str(root / 'tools'))
from source_inventory import source_files

def identity(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}

base.mkdir()
for name in ('model', 'temporary', 'cache'):
    (base / name).mkdir()
shutil.copy2(root / 'build-dev/ernie-image', base / 'ernie-image.snapshot')
shutil.copy2(prior / 'initial.f32', base / 'initial.f32')
shutil.copy2(prior / 'run.py', base / 'run.py')
supervisor = (prior / 'supervisor.py').read_text().replace('ernie-weight-placement512-v1-', 'ernie-weight-cache512-v1-')
(base / 'supervisor.py').write_text(supervisor)
old = json.loads((prior / 'plan.json').read_text())
commands = [[arg.replace(str(prior), str(base)) for arg in command] for command in old['commands']['native']]
commands[0] += ['--dit-cache-mib', '6144', '--ram-reserve-mib', '3072']
baseline = Path('/var/tmp/ernie-runtime512-native-v1')
binding_paths = source_files(root) + [base / name for name in ('ernie-image.snapshot', 'initial.f32', 'run.py', 'supervisor.py')]
binding_paths += [root / 'models/turbo-shared-v2/manifest.json', Path(__file__).resolve()]
binding_paths += [p for p in (baseline / 'native/trace').iterdir() if p.is_file()]
binding_paths += [baseline / 'native/native.png', baseline / 'native/process.json']
binding_paths += [root/'outputs/weight-session-v1/compare.py', prior/'native/process.json', prior/'native/native.png']
binding_paths += [p for p in (prior/'native/trace').iterdir() if p.is_file()]
build = {
    'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'uncommitted_implementation': True,
    'ncnn_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root.parent.parent / 'third_party/ncnn', text=True).strip(),
    'binary': identity(base / 'ernie-image.snapshot'),
    'settings': [line for line in (root / 'build-dev/CMakeCache.txt').read_text().splitlines()
                 if line.startswith(('ERNIE_', 'CMAKE_CXX_COMPILER:', 'CMAKE_BUILD_TYPE:', 'BUILD_TESTING:'))],
}
(base / 'build-identity.json').write_text(json.dumps(build, indent=2) + '\n')
binding_paths += [base / 'build-identity.json']
plan = {
    'scope': 'Native 512x512, same controlled auto-RAM placement plus a 6144MiB prepared host-weight cache and 3072MiB RAM reserve; exact-output and bounded-cache diagnostic, not a paired performance benchmark.',
    'commands': {'native': commands}, 'resource_limits': old['resource_limits'],
    'bindings': {str(p): identity(p) for p in binding_paths}, 'baseline_run': str(baseline),
    'expected_cache': {'budget_bytes': 6144*1024*1024, 'reserve_bytes': 3072*1024*1024, 'minimum_hits': 1, 'total_block_calls': 288},
    'previous_ram_run': str(prior),
    'unchanged_inference': 'same saved initial latent, prompt, 512x512, source32/64DiT slots, native Vector text, 8 Vulkan FP32 steps, CPU direct VAE',
}
(base / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps({'base': str(base), 'binary': build['binary'], 'bindings': len(binding_paths),
                  'plan': identity(base / 'plan.json')}, indent=2))
