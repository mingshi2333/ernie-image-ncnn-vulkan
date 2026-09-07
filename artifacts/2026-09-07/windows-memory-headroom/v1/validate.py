import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
out = Path(__file__).resolve().parent
bindings = json.loads((out / 'source-identity.json').read_text())
assert bindings == {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in bindings}
env = {**os.environ, **json.loads((out / 'wine-environment.json').read_text()), 'OMP_NUM_THREADS': '2'}
records = []
def run(label, args, timeout=180):
    start = time.monotonic()
    with (out / (label + '.log')).open('xb') as log:
        p = subprocess.run(args, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    records.append({'name': label, 'argv': args, 'return_code': p.returncode, 'seconds': time.monotonic() - start})
    (out / 'validation-results.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records[-1]), flush=True)
    return p.returncode
code = run('windows-tests-v2', ['ctest', '--test-dir', str(root / 'build-windows-cpu-v1'),
    '-R', '^(host_memory_cpu|weight_session_cpu)$', '--verbose', '--output-junit', str(out / 'windows-tests-v2.xml')])
if not code:
    code = run('windows-help-v2', ['/usr/bin/wine', str(root / 'build-windows-cpu-v1/ernie-image.exe'), '--help'], 60)
cleanup = run('wine-stop-v2', ['/usr/bin/wineserver', '-k'], 30)
wait = run('wine-exit-v2', ['/usr/bin/wineserver', '-w'], 30)
assert bindings == {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in bindings}
raise SystemExit(code or cleanup or wait)
