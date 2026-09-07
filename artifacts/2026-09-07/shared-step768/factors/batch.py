import json
import os
from pathlib import Path
import subprocess
import sys

base = Path(__file__).resolve().parent
assert not any((base / phase).exists() for phase in ('latent', 'text'))
results = []
(base / 'master-pid.txt').write_text(str(os.getpid()) + '\n')
for phase in ('latent', 'text'):
    with (base / (phase + '-supervisor.log')).open('x') as log:
        result = subprocess.run([sys.executable, str(base / 'supervisor.py'), phase], stdout=log, stderr=subprocess.STDOUT)
    results.append(dict(phase=phase, return_code=result.returncode))
    (base / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results[-1]), flush=True)
    if result.returncode:
        break
raise SystemExit(results[-1]['return_code'])
