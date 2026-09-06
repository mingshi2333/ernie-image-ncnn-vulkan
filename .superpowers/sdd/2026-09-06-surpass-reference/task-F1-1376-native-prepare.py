#!/usr/bin/env python3
"""Prepare identities only. Does not execute a native model."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path.cwd()
base = (ROOT / 'outputs/f1-shape1376-s64-plan-v5').resolve()
out = (ROOT / 'outputs/f1-native-vae1376-preparation-v1').resolve()
out.mkdir()
sys.path.insert(0, str(base / 'source/tools'))
import validate_dit_heads
from vae_reference_scope import runtime_files
from vae_shape_contract import specialize_decoder_graph


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def row(path):
    p = Path(path)
    return {'size_bytes': p.stat().st_size, 'sha256': digest(p)}


def write(name, value):
    (out / name).write_text(json.dumps(value, indent=2) + '\n')

candidate = base / 'vae-candidate'
template = (ROOT / 'models/vae-8x8-v1').resolve()
fixture = validate_dit_heads.verify(candidate)
expected_graph, changes = specialize_decoder_graph((template / 'head.ncnn.param').read_text(), 96, 172, fixed=True)
assert (candidate / 'head.ncnn.param').read_text() == expected_graph and len(changes) == 2
assert digest(template / 'head.ncnn.bin') == digest(candidate / 'head.ncnn.bin') == 'ee7db4b3725079f9baeb8587bcad56fd3f04d92254b90d86acc894c1ec63cd24'
assert digest(candidate / 'fixture.json') == digest(base / 'official-vae/fixture.json')
assert fixture['inputs']['in0']['shape'] == [1,32,96,172]
assert fixture['expected']['out0']['shape'] == [1,3,768,1376]
controller = ROOT / '.superpowers/sdd/2026-09-06-surpass-reference/task-F1-1376-native-controller.py'
shutil.copy2(controller, out / 'controller.py')
shutil.copy2(__file__, out / 'prepare.py')
runner = base / 'head-runner.snapshot'
ldd = subprocess.run(['/usr/bin/ldd', str(runner)], text=True, capture_output=True, check=True)
(out / 'ldd.log').write_text(ldd.stdout + ldd.stderr)
libs = set(re.findall(r'(?:=>\s+|^\s*)(/\S+)', ldd.stdout, re.M))
assert libs and 'not found' not in ldd.stdout
files = {}
source = json.loads((base / 'source-identity.json').read_text())
for name, expected in source['files'].items():
    path = base / 'source' / name
    assert digest(path) == expected
    files[str(path)] = row(path)
for directory in (candidate, template):
    for path in directory.iterdir():
        if path.is_file():
            files[str(path)] = row(path)
for path in (runner, base / 'plan.json', base / 'source-identity.json', base / 'actual-evidence.json',
             base / 'worker-runtime/identity.json', base / 'official-execution/result.json',
             out / 'controller.py', out / 'prepare.py', out / 'ldd.log', Path(sys.executable),
             Path('/usr/bin/time'), Path('/etc/ld.so.cache'), *map(Path, libs)):
    files[str(path)] = row(path)
modules, maps = runtime_files()
for name, info in modules.items():
    files[name] = {'size_bytes': info['size'], 'sha256': info['sha256']}
write('runtime-import.json', {'scope':'Actual validation imports plus ldd resolved native dependencies; not a proof of transient dlopen coverage',
                            'files':modules,'mapped_files':maps,'native_ldd_paths':sorted(libs),
                            'python_prefix':sys.prefix,'python_invocation':os.path.abspath(sys.executable)})
files[str(out/'runtime-import.json')] = row(out/'runtime-import.json')
plan = {'schema_version':1,'status':'prepared_not_executed','native_acceptance_eligible':False,
        'scope':'One fixed CPU decoder component, original FP32 gates; no registry/pipeline/formal claim',
        'unit':'ernie-native-vae1376-v1','python_invocation':os.path.abspath(sys.executable),'python_prefix':sys.prefix,
        'validator':str(base/'source/tools/validate_dit_heads.py'),'candidate':str(candidate),'runner':str(runner),
        'files':files,'changes':changes,'fixture_gates':fixture['gates'],
        'resources':{'memory_max_bytes':16*1024**3,'swap_max_bytes':0,'cpu_affinity':[12,14],
                     'scope_cpu_budget':2,'native_ncnn_threads':4,'host_available_min_bytes':3*1024**3,
                     'outer_timeout_seconds':1800,'validator_runner_timeout_seconds':900,'sample_seconds':.05},
        'outputs':{'in0_shape':[1,32,96,172],'out0_shape':[1,3,768,1376],'out0_elements':3170304,
                   'dtype':'<f4','native_result_path':str(out/'validation/cpu-fp32/result.json')}}
plan['argv']=[plan['python_invocation'],plan['validator'],'--model',plan['candidate'],'--runner',plan['runner'],
              '--output',str(out/'validation'),'--cpu-only','--vae-convolution','direct']
write('plan.json',plan)
launcher=['systemd-run','--user','--scope','--quiet','--unit='+plan['unit'],'-p','MemoryMax=17179869184',
          '-p','MemorySwapMax=0','-p','CPUQuota=200%','taskset','-c','12,14','env','-u','PYTHONPATH',
          'CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2',
          plan['python_invocation'],str(out/'controller.py'),'--plan',str(out/'plan.json'),
          '--plan-sha256',digest(out/'plan.json')]
write('launch.json',{'status':'prepared_not_executed','argv':launcher,
                    'external_env':{'XDG_RUNTIME_DIR':'/run/user/1000','DBUS_SESSION_BUS_ADDRESS':'unix:path=/run/user/1000/bus'},
                    'plan_sha256':digest(out/'plan.json'),'controller_sha256':digest(out/'controller.py'),
                    'requires_root_model_slot':True})
print(json.dumps({'prepared':str(out),'files':len(files),'plan_sha256':digest(out/'plan.json'),
                  'controller_sha256':digest(out/'controller.py'),'launch_sha256':digest(out/'launch.json')}))
