#!/usr/bin/env python3
"""Verify a sealed model, run the C++ block executor, and check its reference gates."""
import argparse
import json
from pathlib import Path
import re
import os
import signal
import subprocess
import numpy as np
from prepare_block import ROOT, sha256
from ncnn_compat import compatible_model_revision


def verify(model_dir, fixture_dir):
    manifest = json.loads((model_dir / 'model.json').read_text())
    fixture = json.loads((fixture_dir / 'fixture.json').read_text())
    if fixture['tokens'] != manifest['tokens'] or fixture['weights_sha256'] != manifest['weights_sha256']:
        raise ValueError('Fixture shape or block weights differ from the static model')
    lock = json.loads((ROOT / 'sources.lock.json').read_text())
    if not compatible_model_revision(manifest.get('ncnn_revision'), lock):
        raise ValueError('Runtime revision differs from the model manifest')
    for name, checksum in manifest['files'].items():
        if Path(name).name != name or sha256(model_dir / name) != checksum:
            raise ValueError(f'Model artifact checksum mismatch: {name}')
    for entry in [*fixture['inputs'].values(), fixture['expected']]:
        if Path(entry['file']).name != entry['file'] or sha256(fixture_dir / entry['file']) != entry['sha256']:
            raise ValueError(f'Fixture checksum mismatch: {entry["file"]}')
        if (fixture_dir / entry['file']).stat().st_size != int(np.prod(entry['shape'])) * 4:
            raise ValueError(f'Fixture byte count mismatch: {entry["file"]}')
    return manifest, fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--runner', type=Path, default=ROOT / 'build/ernie-block-runner')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['cpu', 'vulkan'], default='cpu')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--host-weights', action='store_true')
    parser.add_argument('--device-io', action='store_true')
    parser.add_argument('--trace-attention', action='store_true')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    fixture_dir = args.fixture or args.model
    if args.output.exists():
        parser.error('Use a new result directory to retain previous attempts')
    args.output.mkdir(parents=True)
    summary = {'backend': args.backend, 'precision': args.precision, 'model': str(args.model),
               'fixture': str(fixture_dir), 'passed': False, 'host_weights': args.host_weights,
               'device_io': args.device_io, 'repeat': args.repeat, 'trace_attention': args.trace_attention}
    try:
        manifest, fixture = verify(args.model, fixture_dir)
        summary.update(tokens=fixture['tokens'], scope=fixture['input_scope'],
                       model_manifest_sha256=sha256(args.model / 'model.json'),
                       fixture_sha256=sha256(fixture_dir / 'fixture.json'), runner_sha256=sha256(args.runner),
                       validator_sha256=sha256(__file__))
        output = args.output / 'actual.f32'
        command = [str(args.runner.resolve()), '--model', str(args.model.resolve()), '--fixture', str(fixture_dir.resolve()),
                   '--tokens', str(fixture['tokens']), '--backend', args.backend, '--precision', args.precision,
                   '--repeat', str(args.repeat), '--output', str(output.resolve())]
        if args.host_weights:
            command.append('--host-weights')
        if args.device_io:
            command.append('--device-io')
        if args.trace_attention:
            command.append('--trace-attention')
        summary['command'] = command
        timed_command = ['/usr/bin/time', '-v', '-o', str((args.output / 'resources.log').resolve()), *command]
        with (args.output / 'runner.log').open('w') as log, (args.output / 'gpu-device-memory.log').open('w') as gpu_log:
            sampler = None
            try:
                if args.backend == 'vulkan':
                    try:
                        sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits',
                                                    '--id=0', '--loop-ms=100'], stdout=gpu_log, stderr=subprocess.DEVNULL)
                    except OSError as error:
                        summary['gpu_measurement_unavailable'] = str(error)
                with subprocess.Popen(timed_command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
                    try:
                        return_code = process.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        raise
            finally:
                if sampler is not None:
                    sampler.terminate()
                    sampler.wait(timeout=5)
        samples = [int(line) for line in (args.output / 'gpu-device-memory.log').read_text().splitlines() if line.isdigit()]
        if samples:
            summary['gpu_device_total_mib'] = {'first': samples[0], 'sampled_peak': max(samples), 'samples': len(samples),
                                             'scope': 'Whole NVIDIA device 0, includes other processes; sampled every 100 ms, not allocator/process peak'}
        summary['return_code'] = return_code
        if return_code:
            summary['failure'] = 'Runner failed; inspect runner.log'
        resources = (args.output / 'resources.log').read_text()
        match = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', resources)
        if match:
            summary['max_rss_kib'] = int(match[1])
        for line in (args.output / 'runner.log').read_text().splitlines():
            if line.startswith('{'):
                summary['runtime'] = json.loads(line)
        if return_code == 0:
            ref = np.fromfile(fixture_dir / fixture['expected']['file'], dtype='<f4').astype(np.float64)
            actual = np.fromfile(output, dtype='<f4').astype(np.float64)
            summary['actual_sha256'] = sha256(output)
            if actual.shape != ref.shape or not np.isfinite(actual).all():
                raise ValueError('Invalid shape or non-finite output')
            delta = actual - ref
            maximum = float(np.max(np.abs(delta)))
            nrmse = float(np.linalg.norm(delta) / max(np.linalg.norm(ref), 1e-30))
            limits = fixture['gates'][args.precision]
            # This global-magnitude maximum gate is NOT per-element allclose.
            maximum_limit = limits['atol'] + limits['rtol'] * float(np.max(np.abs(ref)))
            summary.update(max_abs_error=maximum, nrmse=nrmse, max_abs_limit=maximum_limit,
                           nrmse_limit=limits['nrmse'], passed=maximum <= maximum_limit and nrmse <= limits['nrmse'])
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        summary['failure'] = str(error)
    (args.output / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
