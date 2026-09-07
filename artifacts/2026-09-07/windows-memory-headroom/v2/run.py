from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
OUT = Path(__file__).resolve().parent
env = {**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
       'RUSTC': '/home/mingshi/.rustup/toolchains/1.98.0-x86_64-unknown-linux-gnu/bin/rustc',
       'CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER': '/usr/bin/x86_64-w64-mingw32-gcc'}
records = []

def run(name, command, selected_env=env, timeout=600):
    start = time.monotonic()
    with (OUT / (name + '.log')).open('xb') as log:
        result = subprocess.run(command, cwd=ROOT, env=selected_env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    records.append({'name': name, 'argv': command, 'return_code': result.returncode,
                    'seconds': time.monotonic() - start})
    (OUT / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records[-1]), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

paths = ['src/host_memory.cpp', 'src/host_memory.h', 'tests/test_host_memory.cpp', 'docs/RUNNING.md']
binding = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
(OUT / 'source-identity.json').write_text(json.dumps(binding, indent=2) + '\n')
(OUT / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', '--', *paths], cwd=ROOT))
for label, directory in [('windows', 'build-windows-cpu-v1'), ('linux', 'build-install-cpu')]:
    run(label + '-build', ['cmake', '--build', str(ROOT / directory), '--parallel', '2',
                          '--target', 'ernie-host-memory-contract', 'ernie-weight-session-contract', 'ernie-image'])
run('linux-tests', ['ctest', '--test-dir', str(ROOT / 'build-install-cpu'),
                   '-R', '^(host_memory_cpu|weight_session_cpu)$', '--verbose',
                   '--output-junit', str(OUT / 'linux-tests.xml')])
wine_env = {**env, 'WINEPREFIX': '/var/tmp/ernie-windows-headroom-v1/wine-prefix', 'WINEARCH': 'win64',
            'WINEDEBUG': '-all', 'WINEDLLOVERRIDES': 'winemenubuilder.exe=d',
            'WINEPATH': 'Z:\\var\\tmp\\ernie-windows-cpu-v1\\runtime'}
(OUT / 'wine-environment.json').write_text(json.dumps({k: wine_env[k] for k in ('WINEPREFIX', 'WINEARCH', 'WINEDEBUG', 'WINEDLLOVERRIDES', 'WINEPATH')}, indent=2) + '\n')
try:
    run('windows-tests', ['ctest', '--test-dir', str(ROOT / 'build-windows-cpu-v1'),
                         '-R', '^(host_memory_cpu|weight_session_cpu)$', '--verbose',
                         '--output-junit', str(OUT / 'windows-tests.xml')], wine_env, 180)
    run('windows-help', ['/usr/bin/wine', str(ROOT / 'build-windows-cpu-v1/ernie-image.exe'), '--help'], wine_env, 30)
finally:
    run('wine-stop', ['/usr/bin/wineserver', '-k'], wine_env, 30)
    run('wine-exit', ['/usr/bin/wineserver', '-w'], wine_env, 30)
assert binding == {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
