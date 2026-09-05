#!/usr/bin/env python3
"""Validate sealed DiT head graphs against saved official CPU FP32 outputs."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import numpy as np
from prepare_block import ROOT, sha256


def verify(model):
    manifest = json.loads((model / 'model.json').read_text())
    lock = json.loads((ROOT / 'sources.lock.json').read_text())
    fixture = json.loads((model / 'fixture.json').read_text())
    if (manifest['ncnn_revision'] != lock['ncnn']['revision']
            or fixture['official_revision'] != lock['official_model']['revision']
            or manifest['weights'] != fixture['weights']):
        raise ValueError('Source version or weights differ')
    for name, digest in manifest['files'].items():
        if Path(name).name != name or sha256(model / name) != digest:
            raise ValueError(f'Invalid model file: {name}')
    for entry in [*fixture['inputs'].values(), *fixture['expected'].values()]:
        path = model / entry['file']
        if (Path(entry['file']).name != entry['file'] or sha256(path) != entry['sha256']
                or path.stat().st_size != int(np.prod(entry['shape'])) * 4):
            raise ValueError('Invalid tensor fixture')
    return fixture


def run(model, output, runner, backend, precision):
    fixture = verify(model)
    output.mkdir(parents=True)
    command = [str(runner.resolve()), '--model', str(model.resolve()), '--fixture', str(model.resolve()),
               '--output', str((output / 'actual').resolve()), '--component', fixture['component'],
               '--height', str(fixture['height']), '--width', str(fixture['width']),
               '--text-tokens', str(fixture['text_tokens']), '--backend', backend, '--precision', precision]
    result = {'component': fixture['component'], 'backend': backend, 'precision': precision,
              'passed': False, 'command': command, 'runner_sha256': sha256(runner),
              'model_manifest_sha256': sha256(model / 'model.json'), 'validator_sha256': sha256(__file__),
              'scope': fixture['scope'], 'outputs': {}}
    try:
        timed = ['/usr/bin/time', '-v', '-o', str((output / 'resources.log').resolve()), *command]
        with (output / 'runner.log').open('w') as log:
            with subprocess.Popen(timed, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
                try:
                    return_code = process.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
        result['return_code'] = return_code
        if return_code:
            raise ValueError('Runner failed; inspect runner.log')
        for name, entry in fixture['expected'].items():
            ref = np.fromfile(model / entry['file'], '<f4').astype(np.float64)
            actual = np.fromfile(output / 'actual' / f'{name}.f32', '<f4').astype(np.float64)
            if ref.shape != actual.shape or not np.isfinite(actual).all():
                raise ValueError(f'Invalid output tensor: {name}')
            delta = actual - ref
            nrmse = float(np.linalg.norm(delta) / max(np.linalg.norm(ref), 1e-30))
            maximum = float(np.abs(delta).max())
            limits = fixture['gates'][precision]
            max_limit = limits['atol'] + limits['rtol'] * float(np.abs(ref).max())
            result['outputs'][name] = {'nrmse': nrmse, 'max_abs_error': maximum, 'max_abs_limit': max_limit,
                                      'nrmse_limit': limits['nrmse'],
                                      'actual_sha256': sha256(output / 'actual' / f'{name}.f32'),
                                      'passed': nrmse <= limits['nrmse'] and maximum <= max_limit}
        result['passed'] = all(item['passed'] for item in result['outputs'].values())
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        result['failure'] = str(error)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--runner', type=Path, default=ROOT / 'build/ernie-head-runner')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    verify(args.model)
    args.output.mkdir(parents=True)
    runner = args.output / 'ernie-head-runner.snapshot'
    shutil.copy2(args.runner, runner)
    variants = [('cpu', 'fp32')] if args.cpu_only else [('cpu', 'fp32'), ('vulkan', 'fp32'), ('vulkan', 'fp16'), ('vulkan', 'bf16')]
    results = []
    for backend, precision in variants:
        item = run(args.model, args.output / f'{backend}-{precision}', runner, backend, precision)
        results.append(item)
        print(json.dumps({'component': item['component'], 'backend': backend, 'precision': precision,
                          'passed': item['passed'], 'outputs': item['outputs'], 'failure': item.get('failure')}), flush=True)
        (args.output / 'matrix.json').write_text(json.dumps(results, indent=2) + '\n')
    return 0 if all(item['passed'] for item in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
