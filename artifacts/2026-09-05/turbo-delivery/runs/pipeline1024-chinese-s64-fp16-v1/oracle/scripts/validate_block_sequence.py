#!/usr/bin/env python3
"""Validate connected real-weight blocks with streamed and bounded resident weights."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import numpy as np
import torch
from export_dit_block import load_block, make_inputs, save_tensor
from prepare_block import ROOT, sha256
from validate_dit_block import verify


def reference(models, output):
    verified = [verify(path, path) for path in models]
    base = verified[0][1]
    if any(item[0]['tokens'] != base['tokens'] for item in verified):
        raise ValueError('Models have different static token counts')
    blocks = [item[0]['block'] for item in verified]
    if blocks != list(range(blocks[0], blocks[0] + len(blocks))):
        raise ValueError('Select consecutive block indices in order')
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    inputs, freqs = make_inputs(*base['grid'], base['text_tokens'], base['valid_text'], base['seed'])
    output.mkdir(parents=True)
    entries = {f'in{i}': save_tensor(output / f'in{i}.f32', value) for i, value in enumerate(inputs)}
    if any(item['sha256'] != base['inputs'][name]['sha256'] for name, item in entries.items()):
        raise ValueError('This reference generator requires the saved synthetic fixture family')
    state = {'tokens': base['tokens'], 'blocks': blocks, 'inputs': entries,
             'scope': 'Connected real block weights with synthetic initial activation and shared AdaLN; no preprocessor or finalizer',
             'model_manifests': [sha256(path / 'model.json') for path in models],
             'source_sha256': sha256(__file__), 'torch': torch.__version__, 'reference_stages': [],
             'gates': {'fp32': {'atol': .0002, 'rtol': .0002, 'nrmse': .0002},
                       'fp16': {'atol': .03, 'rtol': .03, 'nrmse': .03},
                       'bf16': {'atol': .2, 'rtol': .2, 'nrmse': .12}}}
    current = inputs[0].transpose(0, 1)
    temb = list(inputs[1:7])
    for index, (manifest, fixture) in zip(blocks, verified):
        started = time.perf_counter()
        weights = ROOT / f'models/official/dit-block-{index:02d}.safetensors'
        block, component = load_block(weights, index)
        if component['sha256'] != manifest['weights_sha256']:
            raise ValueError('Reference weights differ from converted model')
        current = block(current, freqs, temb, attention_mask=inputs[-1][None, None])
        entry = save_tensor(output / f'after-block-{index:02d}.f32', current.transpose(0, 1))
        state['reference_stages'].append({'block': index, 'output': entry,
                                          'elapsed_seconds': time.perf_counter() - started})
        del block
    state['expected'] = state['reference_stages'][-1]['output']
    (output / 'fixture.json').write_text(json.dumps(state, indent=2) + '\n')
    return state


def run(models, fixture_dir, fixture, output, runner, backend, precision, policy, extra_args=()):
    output.mkdir(parents=True)
    command = [str(runner.resolve()), '--fixture', str(fixture_dir.resolve()), '--tokens', str(fixture['tokens']),
               '--output', str((output / 'actual.f32').resolve()), '--backend', backend,
               '--precision', precision, '--policy', policy, *extra_args]
    for model in models:
        command += ['--model', str(model.resolve())]
    result = {'backend': backend, 'precision': precision, 'policy': policy, 'passed': False,
              'command': command, 'runner_sha256': sha256(runner), 'fixture_sha256': sha256(fixture_dir / 'fixture.json')}
    try:
        timed = ['/usr/bin/time', '-v', '-o', str((output / 'resources.log').resolve()), *command]
        with (output / 'runner.log').open('w') as log, (output / 'gpu-device-memory.log').open('w') as gpu_log:
            sampler = None
            try:
                if backend == 'vulkan':
                    try:
                        sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits',
                                                    '--id=0', '--loop-ms=100'], stdout=gpu_log, stderr=subprocess.DEVNULL)
                    except OSError as error:
                        result['gpu_measurement_unavailable'] = str(error)
                with subprocess.Popen(timed, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
                    try:
                        return_code = process.wait(timeout=1800)
                    except subprocess.TimeoutExpired:
                        # GNU time is a parent process: terminate the whole group so
                        # a timed-out GPU runner cannot continue holding its weights.
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        raise
            finally:
                if sampler is not None:
                    sampler.terminate()
                    sampler.wait(timeout=5)
        samples = [int(line) for line in (output / 'gpu-device-memory.log').read_text().splitlines() if line.isdigit()]
        if samples:
            result['gpu_device_total_mib'] = {'first': samples[0], 'sampled_peak': max(samples), 'samples': len(samples),
                                             'scope': 'Whole NVIDIA device 0 including other processes; 100 ms samples, not allocator/process peak'}
        result['return_code'] = return_code
        if return_code:
            raise ValueError('Sequence runner failed; inspect runner.log')
        result['runtime'] = [json.loads(line) for line in (output / 'runner.log').read_text().splitlines()
                             if line.startswith('{')][-1]
        resources = (output / 'resources.log').read_text()
        match = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', resources)
        if match:
            result['max_rss_kib'] = int(match[1])
        expected_path = fixture_dir / fixture['expected']['file']
        if sha256(expected_path) != fixture['expected']['sha256']:
            raise ValueError('Reference checksum changed')
        expected = np.fromfile(expected_path, '<f4').astype(np.float64)
        actual = np.fromfile(output / 'actual.f32', '<f4').astype(np.float64)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError('Invalid sequence output')
        delta = actual - expected
        limits = fixture['gates'][precision]
        maximum_limit = limits['atol'] + limits['rtol'] * float(np.abs(expected).max())
        nrmse = float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-30))
        maximum = float(np.abs(delta).max())
        result.update(nrmse=nrmse, max_abs_error=maximum, max_abs_limit=maximum_limit,
                      nrmse_limit=limits['nrmse'], actual_sha256=sha256(output / 'actual.f32'),
                      passed=nrmse <= limits['nrmse'] and maximum <= maximum_limit)
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as error:
        result['failure'] = str(error)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runner', type=Path, default=ROOT / 'build/ernie-block-sequence-runner')
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--isolated-pipeline-cache', action='store_true', help='Retain the old per-Net cache baseline')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    fixture_dir = args.output / 'reference'
    fixture = reference(args.model, fixture_dir)
    variants = [('cpu', 'fp32')] if args.cpu_only else [('cpu', 'fp32'), ('vulkan', 'fp32'), ('vulkan', 'fp16'), ('vulkan', 'bf16')]
    policies = ['stream', 'resident'] if len(args.model) <= 2 else ['stream']
    results = []
    for backend, precision in variants:
        previous = None
        for policy in policies:
            result = run(args.model, fixture_dir, fixture, args.output / f'{backend}-{precision}-{policy}',
                         args.runner, backend, precision, policy,
                         ['--isolated-pipeline-cache'] if args.isolated_pipeline_cache else [])
            if previous and result['passed'] and previous['passed']:
                result['matches_stream_exactly'] = result['actual_sha256'] == previous['actual_sha256']
                result['passed'] &= result['matches_stream_exactly']
            previous = result
            results.append(result)
            print(json.dumps({key: result.get(key) for key in ['backend', 'precision', 'policy', 'passed', 'nrmse', 'max_rss_kib']}), flush=True)
            (args.output / 'matrix.json').write_text(json.dumps(results, indent=2) + '\n')
    return 0 if all(result['passed'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
