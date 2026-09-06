#!/usr/bin/env python3
"""Prepare or explicitly execute one frozen Chinese step6 time-feature swap.

Preparation is CPU/file-only. Execution is an implementation-difference diagnostic,
not a free-running acceptance run. No reference gates are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

INPUTS = {'in0', 'in1', 'in2', 'in3', 'in4', 'in5'}
RUNNER_SHA = 'a4a80b564042d4020096edf947d30bd28a5fdf354005d1c17685fddddcdbe1fc'
REFERENCE_SHA = '81a853d9db2c6aba80a3fa72abac618dd9196f8598b5055cc2a41586480580b7'
PACKAGE_SHA = '72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1'


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def tensor(root, entry):
    name = entry['file']
    shape = entry['shape']
    if (not isinstance(name, str) or Path(name).name != name or name in ('.', '..')
            or not isinstance(shape, list) or not shape
            or any(type(x) is not int or x < 1 for x in shape)):
        raise ValueError('Invalid tensor entry')
    count = 1
    for x in shape:
        count *= x
    path = root/name
    if count > 32*1024*1024 or path.stat().st_size != count*4 or sha(path) != entry['sha256']:
        raise ValueError('Tensor identity/size differs')
    if not np.isfinite(np.fromfile(path, '<f4')).all():
        raise ValueError('Nonfinite tensor')
    return path


def swap_inputs(base, native, output):
    """Validate all inputs before creating a new fixture; change only in2."""
    fixture = json.loads((base/'fixture.json').read_text())
    if set(fixture['inputs']) != INPUTS:
        raise ValueError('Require exactly six inputs')
    entries = {**fixture['inputs'], 'expected': fixture['expected']}
    source = {k: tensor(base, e) for k, e in entries.items()}
    if len({e['file'] for e in entries.values()}) != len(entries):
        raise ValueError('Tensor filenames must be distinct')
    if native.stat().st_size != 4096*4 or not np.isfinite(np.fromfile(native, '<f4')).all():
        raise ValueError('Native feature requires 4096 finite FP32 values')
    if entries['in2']['shape'] not in ([4096], [1, 4096]):
        raise ValueError('Unexpected time-feature shape')
    if output.exists():
        raise ValueError('Use a new fixture directory')
    output.mkdir(parents=True)
    for k, path in source.items():
        shutil.copy2(native if k == 'in2' else path, output/entries[k]['file'])
    fixture['scope'] = 'Single time-feature implementation swap; all other official inputs identical; not free-running acceptance'
    fixture['native_acceptance_eligible'] = False
    fixture['time_feature_swap'] = {'official_sha256': entries['in2']['sha256'], 'native_sha256': sha(native)}
    fixture['inputs']['in2']['sha256'] = sha(native)
    (output/'fixture.json').write_text(json.dumps(fixture, indent=2)+'\n')
    return fixture


def prepare(args):
    base, execution, native, output = [x.resolve() for x in (args.baseline, args.execution, args.native_features, args.output)]
    if output.exists():
        raise ValueError('Use a new output directory')
    fixture = json.loads((base/'fixture/fixture.json').read_text())
    result = json.loads((base/'native/result.json').read_text())
    if (fixture['step'] != 6 or fixture['reference_fixture_sha256'] != REFERENCE_SHA
            or fixture['package_manifest_sha256'] != PACKAGE_SHA or result['passed'] is not True
            or result['runner_sha256'] != RUNNER_SHA or sha(execution/'runner.snapshot') != RUNNER_SHA
            or result['fixture_sha256'] != sha(base/'fixture/fixture.json')):
        raise ValueError('Require the reviewed same-input baseline/runner')
    snapshot = json.loads((execution/'snapshot.json').read_text())
    for name, digest in snapshot['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(execution/name) != digest:
            raise ValueError('Execution source snapshot changed')
    fixture = swap_inputs(base/'fixture', native, output/'fixture')
    command = list(result['command'])
    command[0] = str(execution/'runner.snapshot')
    command[command.index('--fixture')+1] = str(output/'fixture')
    command[command.index('--output')+1] = str(output/'native/actual.f32')
    bound = [base/'fixture/fixture.json', base/'native/result.json', execution/'snapshot.json',
             execution/'runner.snapshot', native, Path(__file__).resolve(), output/'fixture/fixture.json']
    plan = {'scope': fixture['scope'], 'status': 'prepared_not_executed', 'native_acceptance_eligible': False,
            'baseline': str(base), 'execution': str(execution), 'output': str(output),
            'command': command, 'bound_sha256': {str(p): sha(p) for p in bound}}
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps({'status': plan['status'], 'plan': str(output/'plan.json'), 'command': command}))


def execute(plan_path):
    plan = json.loads(plan_path.read_text())
    for name, digest in plan['bound_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Prepared input/tool identity changed')
    execution, output = Path(plan['execution']), Path(plan['output'])
    snapshot = json.loads((execution/'snapshot.json').read_text())
    for name, digest in snapshot['files'].items():
        if sha(execution/name) != digest:
            raise ValueError('Frozen execution source changed')
    fixture = json.loads((output/'fixture/fixture.json').read_text())
    for entry in [*fixture['inputs'].values(), fixture['expected']]:
        tensor(output/'fixture', entry)
    if sha(execution/'runner.snapshot') != RUNNER_SHA:
        raise ValueError('Unexpected prediction runner')
    sys.path.insert(0, str(execution/'tools'))
    from validate_block_sequence import run
    command = plan['command']
    original = json.loads((Path(plan['baseline'])/'native/result.json').read_text())['command']
    expected = list(original)
    expected[0] = str(execution/'runner.snapshot')
    expected[expected.index('--fixture')+1] = str(output/'fixture')
    expected[expected.index('--output')+1] = str(output/'native/actual.f32')
    if command != expected:
        raise ValueError('Prepared command differs beyond the single fixture/output substitution')
    from package_model import verify_package
    package = Path(command[command.index('--input-head')+1]).parent.parent
    verify_package(package)
    if sha(package/'manifest.json') != PACKAGE_SHA:
        raise ValueError('Prediction package differs')
    models = [Path(command[i+1]) for i, token in enumerate(command) if token == '--model']
    extra = []
    for key in ('--input-head', '--output-head', '--width', '--height', '--text-tokens'):
        extra += [key, command[command.index(key)+1]]
    result = run(models, output/'fixture', fixture, output/'native', execution/'runner.snapshot',
                 'vulkan', 'fp32', 'stream', extra_args=extra)
    print(json.dumps({k: result.get(k) for k in ('passed', 'failure', 'nrmse', 'max_abs_error')}))
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', type=Path, help='Explicitly run a prepared plan; needs exclusive GPU and external resource guard')
    for flag in ('baseline', 'execution', 'native-features', 'output'):
        parser.add_argument('--'+flag, type=Path)
    args = parser.parse_args()
    if args.execute:
        if any(getattr(args, k) is not None for k in ('baseline', 'execution', 'native_features', 'output')):
            parser.error('Execution cannot be combined with preparation')
        return execute(args.execute)
    if any(getattr(args, k) is None for k in ('baseline', 'execution', 'native_features', 'output')):
        parser.error('Preparation requires baseline, execution, native-features and output')
    prepare(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
