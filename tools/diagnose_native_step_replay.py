#!/usr/bin/env python3
"""Freeze and replay the actual saved native Chinese step6 call.

This is native-call reproducibility, not comparison to an official oracle.
Exact bytes are the replay criterion; no official quality gates are imported.
Preparation performs no GPU work. --execute needs an external resource guard.
"""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

SOURCE_RUNNER = '7d9ea0d6dd6633781f17671bc5c4fa2f8fbf01d5b88d6e2c114e21175168b06c'
REPLAY_RUNNER = 'a4a80b564042d4020096edf947d30bd28a5fdf354005d1c17685fddddcdbe1fc'
PACKAGE = '72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1'
REFERENCE = '81a853d9db2c6aba80a3fa72abac618dd9196f8598b5055cc2a41586480580b7'
FEATURE = '5919ea0805bf18fa0de6fcb1ebf080cd7e62af72ec411a57f5f9dce71cd604c1'


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def checked(path, shape, digest):
    if not isinstance(shape, list) or not shape or any(type(x) is not int or x < 1 for x in shape):
        raise ValueError('Invalid shape')
    count = 1
    for x in shape:
        count *= x
    if count > 32*1024*1024 or path.stat().st_size != count*4 or sha(path) != digest:
        raise ValueError('Saved tensor identity/size differs')
    value = np.fromfile(path, '<f4')
    if not np.isfinite(value).all():
        raise ValueError('Saved tensor is nonfinite')
    return value


def compare(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Invalid replay output')
    a, b = actual.astype('f8'), expected.astype('f8')
    delta = a-b
    return {'bitwise_equal': actual.tobytes() == expected.tobytes(),
            'max_abs_error': float(np.abs(delta).max()),
            'nrmse': float(np.linalg.norm(delta)/max(np.linalg.norm(b), 1e-30)),
            'different_float_count': int(np.count_nonzero(actual != expected))}


def native_command(execution, output, package):
    command = [str(execution/'runner.snapshot'), '--fixture', str(output/'fixture'), '--tokens', '4160',
               '--output', str(output/'native/actual.f32'), '--backend', 'vulkan', '--precision', 'fp32',
               '--policy', 'stream', '--input-head', str(package/'dit/input'), '--output-head', str(package/'dit/output'),
               '--width', '64', '--height', '64', '--text-tokens', '64']
    for i in range(36):
        command += ['--model', str(package/f'dit/block-{i:02d}')]
    return command


def prepare(args):
    source, execution, feature, output = [p.resolve() for p in (args.source, args.execution, args.native_features, args.output)]
    if output.exists():
        raise ValueError('Use a new output directory')
    result = json.loads((source/'result.json').read_text())
    reference = json.loads((source/'reference/fixture.json').read_text())
    cfg = reference['config']
    if (result['runner_sha256'] != SOURCE_RUNNER or sha(source/'ernie-image.snapshot') != SOURCE_RUNNER
            or result['package_manifest_sha256'] != PACKAGE or result['reference_fixture_sha256'] != REFERENCE
            or sha(source/'reference/fixture.json') != REFERENCE or cfg['packed_width'] != 64
            or cfg['packed_height'] != 64 or cfg['text_bucket'] != 64 or cfg['dit_text_tokens'] != 64
            or reference['steps'] != 8 or result['device'] != 'vulkan' or result['dit_precision'] != 'fp32'
            or sha(execution/'runner.snapshot') != REPLAY_RUNNER or sha(feature) != FEATURE):
        raise ValueError('Unexpected source run, shape, feature or replay runner')
    package = Path(result['command'][result['command'].index('--model')+1])
    if sha(package/'manifest.json') != PACKAGE:
        raise ValueError('Package identity differs')
    rows = {x['tensor']: x for x in result['comparisons']}
    if len(rows) != 25 or len(result['comparisons']) != 25:
        raise ValueError('Source comparisons are incomplete/duplicate')
    entries = {
        'in0': ('step-5', [128,64,64]), 'in1': ('padded-text', [64,3072]),
        'in3': ('constant-0', [4160,128]), 'in4': ('constant-1', [4160,128]),
        'in5': ('constant-2', [4160,4160]), 'expected': ('prediction-6', [128,64,64]),
    }
    prepared = {}
    comparison = {}
    for name, (trace_name, shape) in entries.items():
        path = source/'trace'/(trace_name+'.f32')
        native = checked(path, shape, rows[trace_name]['sha256'])
        prepared[name] = {'source': str(path), 'shape': shape, 'sha256': sha(path), 'dtype': 'float32_le'}
        if trace_name in reference['inputs']:
            item = reference['inputs'][trace_name]
            official = checked(source/'reference'/item['file'], item['shape'], item['sha256'])
            comparison[trace_name] = compare(native, official)
    checked(feature, [4096], FEATURE)
    prepared['in2'] = {'source': str(feature), 'shape': [4096], 'sha256': FEATURE, 'dtype': 'float32_le'}
    snapshot = json.loads((execution/'snapshot.json').read_text())
    for name, digest in snapshot['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(execution/name) != digest:
            raise ValueError('Prediction source snapshot differs')
    fixture_dir = output/'fixture'; fixture_dir.mkdir(parents=True)
    for name, entry in prepared.items():
        entry['file'] = name+'.f32';shutil.copy2(entry['source'], fixture_dir/entry['file'])
    command = native_command(execution, output, package)
    plan = {'protocol': 'native-chinese-step6-replay-v1', 'status': 'prepared_not_executed',
            'scope': 'Reproduce saved native prediction-6 from all six actual native inputs; no official acceptance gate',
            'native_acceptance_eligible': False, 'replay_criterion': 'exact FP32 output bytes',
            'source': str(source), 'execution': str(execution), 'output': str(output), 'command': command,
            'source_runner_sha256': SOURCE_RUNNER, 'replay_runner_sha256': REPLAY_RUNNER,
            'cross_binary_replay': True, 'package_manifest_sha256': PACKAGE,
            'step': 6, 'timestep': 571.4285888671875, 'inputs': {k:v for k,v in prepared.items() if k != 'expected'},
            'expected': prepared['expected'], 'native_vs_official_conditioning': comparison,
            'bound_sha256': {str(p): sha(p) for p in [source/'result.json', source/'reference/fixture.json',
                execution/'snapshot.json', package/'manifest.json', Path(__file__).resolve()]}}
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps({'plan': str(output/'plan.json'), 'status': plan['status'], 'native_vs_official': comparison}))


def execute(path):
    plan = json.loads(path.read_text());output = Path(plan['output']);execution = Path(plan['execution'])
    if plan['protocol'] != 'native-chinese-step6-replay-v1' or plan.get('native_acceptance_eligible') is not False:
        raise ValueError('Not a native replay plan')
    for p, digest in plan['bound_sha256'].items():
        if sha(p) != digest:raise ValueError('Bound input/tool changed')
    for name, digest in json.loads((execution/'snapshot.json').read_text())['files'].items():
        if sha(execution/name) != digest:raise ValueError('Frozen prediction sources changed')
    if sha(execution/'runner.snapshot') != REPLAY_RUNNER:raise ValueError('Unexpected runner')
    for entry in [*plan['inputs'].values(), plan['expected']]:
        checked(output/'fixture'/entry['file'], entry['shape'], entry['sha256'])
    sys.path.insert(0, str(execution/'tools'))
    from package_model import verify_package
    command = plan['command'];package = Path(command[command.index('--input-head')+1]).parent.parent
    verify_package(package)
    if sha(package/'manifest.json') != PACKAGE:raise ValueError('Unreviewed package')
    if command != native_command(execution, output, package):raise ValueError('Command layout/options differ')
    native = output/'native';native.mkdir()
    with (native/'runner.log').open('w') as log:
        process = subprocess.Popen(['/usr/bin/time','-v','-o',str(native/'resources.log'),*command],
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
            (native/'timeout.json').write_text(json.dumps({'status':'timeout','scope':'whole native process group killed'})+'\n')
    result = {'protocol': plan['protocol'], 'native_acceptance_eligible': False,
              'return_code': process.returncode, 'plan_sha256': sha(path), 'runner_sha256': REPLAY_RUNNER}
    if process.returncode == 0:
        actual_path = native/'actual.f32';actual_digest = sha(actual_path)
        actual = checked(actual_path, plan['expected']['shape'], actual_digest)
        expected = checked(output/'fixture'/plan['expected']['file'], plan['expected']['shape'], plan['expected']['sha256'])
        result.update(compare(actual,expected));result['actual_sha256']=actual_digest
        result['status'] = 'reproduced_exactly' if result['bitwise_equal'] else 'completed_with_replay_difference'
    else:result['status']='native_execution_failed'
    (native/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    return 0 if result.get('bitwise_equal') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', type=Path)
    for name in ('source','execution','native-features','output'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    supplied=[getattr(args,k) for k in ('source','execution','native_features','output')]
    if args.execute:
        if any(p is not None for p in supplied):parser.error('Execution and preparation cannot be combined')
        return execute(args.execute)
    if any(p is None for p in supplied):parser.error('Provide all preparation paths')
    prepare(args);return 0


if __name__=='__main__':raise SystemExit(main())
