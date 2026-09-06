#!/usr/bin/env python3
"""Prepare one-factor Chinese step6 diagnostics; execute only under an external GPU guard.

No official quality gates: official prediction distance is descriptive because
these hybrid inputs do not equal the full official input tuple.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import numpy as np
from diagnose_native_step_replay import sha, checked, compare, native_command, PACKAGE, REFERENCE, REPLAY_RUNNER

PROTOCOL = 'native-chinese-step6-single-factor-v1'
BASE_PLAN = '78700f98092646f3f86d9b19ec91d22537fabb9d1f00bbeafd0e2a4a3e08a0d3'
BASE_RESULT = '70083c0574432a2ea7c45c85dac2951fa486b60d4bfd1a24eceddb1811e729f0'
FACTORS = {'latent': ('in0', '8368a393b89407618726df770f3015892626762e90a986bc851d32d7a85d48f4'),
           'text': ('in1', '40c6b63478d16f90370925bfac16381661bef12a48ae197b59bdc5ae1a1f316d')}
ORACLE = '4dac3a98973693a7c94438f8f9760411eabedd239736e85744d73b922a5dc127'


def validate_factor(base, inputs, factor):
    if factor not in FACTORS or set(inputs) != set(base) or set(inputs) != {f'in{i}' for i in range(6)}:
        raise ValueError('Invalid factor/input denominator')
    name, digest = FACTORS[factor]
    for key, original in base.items():
        candidate = inputs[key]
        if candidate['shape'] != original['shape'] or candidate['dtype'] != 'float32_le':
            raise ValueError('Shape/dtype changed')
        expected = digest if key == name else original['sha256']
        if candidate['sha256'] != expected:
            raise ValueError('Not the exact declared single-factor replacement')
    if inputs[name]['sha256'] == base[name]['sha256']:
        raise ValueError('Selected factor did not change')


def describe(actual, native, oracle):
    return {'difference_from_native_replay': compare(actual, native),
            'descriptive_distance_from_official_prediction': compare(actual, oracle),
            'native_acceptance_eligible': False, 'official_gate_applied': False,
            'interpretation': 'Conditional hybrid-input sensitivity; not pure implementation rounding error'}


def baseline(path):
    if sha(path) != BASE_PLAN or sha(path.parent/'native/result.json') != BASE_RESULT:
        raise ValueError('Unreviewed or unsuccessful baseline')
    p = json.loads(path.read_text())
    r = json.loads((path.parent/'native/result.json').read_text())
    if not r['bitwise_equal'] or r['status'] != 'reproduced_exactly':
        raise ValueError('Baseline must reproduce the native bytes')
    checked(path.parent/'native/actual.f32', p['expected']['shape'], p['expected']['sha256'])
    return p


def prepare(base_path, factor, output):
    base_path, output = base_path.resolve(), output.resolve()
    if output.exists(): raise ValueError('Use a fresh output directory')
    base = baseline(base_path)
    reference = Path(base['source'])/'reference'
    if sha(reference/'fixture.json') != REFERENCE: raise ValueError('Unreviewed oracle')
    fixture = json.loads((reference/'fixture.json').read_text())
    item = fixture['outputs'][5]['step'] if factor == 'latent' else fixture['inputs']['padded-text']
    name, digest = FACTORS[factor]
    if item['sha256'] != digest or fixture['outputs'][6]['prediction']['sha256'] != ORACLE:
        raise ValueError('Wrong official boundary')
    inputs = copy.deepcopy(base['inputs'])
    inputs[name].update(source=str(reference/item['file']), sha256=digest)
    validate_factor(base['inputs'], inputs, factor)
    oracle = {'source': str(reference/fixture['outputs'][6]['prediction']['file']),
              'shape': [128,64,64], 'dtype':'float32_le', 'sha256':ORACLE, 'file':'official-prediction.f32'}
    native = dict(base['expected'], file='native-prediction.f32')
    for entry in [*inputs.values(), native, oracle]:
        checked(Path(entry['source']), entry['shape'], entry['sha256'])
    (output/'fixture').mkdir(parents=True)
    for entry in [*inputs.values(), native, oracle]:shutil.copy2(entry['source'],output/'fixture'/entry['file'])
    execution = Path(base['execution']);package = Path(base['command'][base['command'].index('--input-head')+1]).parent.parent
    plan = {'protocol':PROTOCOL,'status':'prepared_not_executed','factor':factor,
            'native_acceptance_eligible':False,'official_gate_applied':False,
            'base_plan':str(base_path),'base_plan_sha256':BASE_PLAN,'base_result_sha256':BASE_RESULT,
            'inputs':inputs,'native_prediction':native,'official_prediction':oracle,
            'output':str(output),'execution':str(execution),'command':native_command(execution,output,package),
            'bound_sha256':{**base['bound_sha256'],str(reference/'fixture.json'):REFERENCE,
                            str(Path(__file__).resolve()):sha(__file__)}}
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


def execute(path):
    p=json.loads(path.read_text());out=Path(p['output']);execution=Path(p['execution'])
    if p['protocol'] != PROTOCOL or p['native_acceptance_eligible'] is not False or p['official_gate_applied'] is not False:
        raise ValueError('Wrong diagnostic contract')
    base=baseline(Path(p['base_plan']));validate_factor(base['inputs'],p['inputs'],p['factor'])
    for source,digest in p['bound_sha256'].items():
        if sha(source)!=digest:raise ValueError('Bound source changed')
    for name,digest in json.loads((execution/'snapshot.json').read_text())['files'].items():
        if sha(execution/name)!=digest:raise ValueError('Frozen source changed')
    if sha(execution/'runner.snapshot')!=REPLAY_RUNNER:raise ValueError('Runner identity differs')
    if p['native_prediction']['sha256']!=base['expected']['sha256'] or p['official_prediction']['sha256']!=ORACLE:
        raise ValueError('Prediction comparator changed')
    for entry in [*p['inputs'].values(),p['native_prediction'],p['official_prediction']]:
        checked(out/'fixture'/entry['file'],entry['shape'],entry['sha256'])
    command=p['command'];package=Path(command[command.index('--input-head')+1]).parent.parent
    if command!=native_command(execution,out,package) or sha(package/'manifest.json')!=PACKAGE:
        raise ValueError('Command/package changed')
    sys.path.insert(0,str(execution/'tools'))
    from package_model import verify_package
    verify_package(package)
    native=out/'native';native.mkdir()
    with (native/'runner.log').open('w') as log:
        process=subprocess.Popen(['/usr/bin/time','-v','-o',str(native/'resources.log'),*command],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
            (native/'timeout.json').write_text(json.dumps({'status':'timeout','scope':'whole native process group killed'})+'\n')
    r={'protocol':PROTOCOL,'factor':p['factor'],'plan_sha256':sha(path),'runner_sha256':REPLAY_RUNNER,
       'native_acceptance_eligible':False,'official_gate_applied':False,'return_code':process.returncode,
       'status':'native_execution_failed'}
    if process.returncode==0:
        actual=native/'actual.f32';a=checked(actual,[128,64,64],sha(actual))
        n=checked(out/'fixture'/p['native_prediction']['file'],[128,64,64],base['expected']['sha256'])
        o=checked(out/'fixture'/p['official_prediction']['file'],[128,64,64],ORACLE)
        r.update(describe(a,n,o));r.update(actual_sha256=sha(actual),status='completed_descriptive_diagnostic')
    (native/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
    return 0 if process.returncode==0 else 1


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--execute',type=Path)
    a.add_argument('--baseline',type=Path);a.add_argument('--factor',choices=FACTORS);a.add_argument('--output',type=Path);args=a.parse_args()
    if args.execute:
        if args.baseline or args.factor or args.output:a.error('Do not combine prepare/execute')
        return execute(args.execute)
    if not all((args.baseline,args.factor,args.output)):a.error('Require baseline/factor/output')
    p=prepare(args.baseline,args.factor,args.output);print(json.dumps({'status':p['status'],'factor':p['factor']}));return 0

if __name__=='__main__':raise SystemExit(main())
