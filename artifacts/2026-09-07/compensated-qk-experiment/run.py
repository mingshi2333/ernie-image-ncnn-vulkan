"""Run the fixed OFF/ON operator checks and one identical real teacher block."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

base=Path(__file__).resolve().parent
phase, expected_plan=sys.argv[1:]
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
assert sha(base/'plan.json')==expected_plan
plan=json.loads((base/'plan.json').read_text())
assert phase in ('off','on')
def verify():
    for name,item in plan['bindings'].items():
        path=Path(name)
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('Frozen binding changed: '+name)
verify()
results=[]
for i,command in enumerate(plan['commands'][phase]):
    with (base/phase/f'command-{i}.log').open('xb') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    results.append({'command':command,'return_code':result.returncode})
    if result.returncode:
        break
verify()
(base/phase/'worker-result.json').write_text(json.dumps(results,indent=2)+'\n')
raise SystemExit(results[-1]['return_code'])
