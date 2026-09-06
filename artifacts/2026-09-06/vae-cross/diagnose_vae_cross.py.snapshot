#!/usr/bin/env python3
"""Cross official/native decoders and saved latents without rerunning the DiT.

This diagnoses a saved run. Passing a crossed decoder does not establish full
pipeline parity, and the four finite differences are not causal percentages.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_FP32 = {'nrmse': .003, 'global_rtol': .01, 'atol': .0002}
VAE_FP32 = {'nrmse': 2e-5, 'global_rtol': .0002, 'atol': .0002}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def finite_arrays(*values):
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    if not arrays or not arrays[0].size:
        raise ValueError('Require nonempty tensors')
    if any(value.shape != arrays[0].shape or not np.isfinite(value).all()
           for value in arrays):
        raise ValueError('Tensor shapes differ or values are nonfinite')
    return arrays


def compare_arrays(reference, candidate, gates):
    reference, candidate = finite_arrays(reference, candidate)
    delta = candidate - reference
    absolute = np.abs(delta)
    maximum = float(absolute.max())
    norm = float(np.linalg.norm(delta.ravel()) /
                 max(np.linalg.norm(reference.ravel()), 1e-30))
    limit = gates['atol'] + gates['global_rtol'] * float(np.abs(reference).max())
    return {'max_abs_error': maximum, 'nrmse': norm,
            'max_abs_limit': limit, 'gates': dict(gates),
            'max_index': [int(i) for i in np.unravel_index(absolute.argmax(), delta.shape)],
            'passed': bool(norm <= gates['nrmse'] and maximum <= limit)}


def cross_terms(oo, no, on, nn):
    oo, no, on, nn = finite_arrays(oo, no, on, nn)
    differences = {'decoder_on_official': no - oo, 'decoder_on_native': nn - on,
                   'input_through_official': on - oo, 'input_through_native': nn - no,
                   'connected': nn - oo, 'interaction': nn - on - no + oo}
    return {name + '_max': float(np.abs(value).max())
            for name, value in differences.items()}


def load_tensor(path, shape, expected_sha256=None):
    path = Path(path)
    if path.stat().st_size != int(np.prod(shape)) * 4:
        raise ValueError(f'Tensor byte count differs: {path}')
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError(f'Tensor checksum differs: {path}')
    value = np.fromfile(path, '<f4').reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError(f'Nonfinite tensor: {path}')
    return value


def run_child(command, directory, timeout):
    directory.mkdir(parents=True)
    record = {'command': command, 'status': 'running'}
    started = time.perf_counter()
    environment = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    with (directory/'runner.log').open('w') as log:
        with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True, env=environment) as process:
            try:
                return_code = process.wait(timeout=timeout)
                record.update(status='completed' if return_code == 0 else 'failed',
                              return_code=return_code)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                record.update(status='timeout', return_code=process.returncode)
    record['elapsed_seconds'] = time.perf_counter() - started
    (directory/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    if record['status'] != 'completed':
        raise RuntimeError(f'Crossed decoder failed: {directory}; inspect runner.log')
    return record


def official_decode(input_path, output_path, height, width):
    import inspect
    import torch
    from diffusers import AutoencoderKLFlux2
    from export_vae import load_vae
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, manifests = load_vae()
    latent = torch.from_numpy(load_tensor(input_path, (1, 32, height, width)))
    decoded = model._decode(latent, return_dict=False)[0]
    value = decoded.detach().cpu().numpy().astype('<f4')
    if not np.isfinite(value).all():
        raise ValueError('Official decoder produced nonfinite output')
    with Path(output_path).open('xb') as stream:
        value.tofile(stream)
    print(json.dumps({'torch': torch.__version__, 'device': 'cpu', 'dtype': 'fp32',
                      'official_class_sha256': sha256(inspect.getfile(AutoencoderKLFlux2)),
                      'weights': {k: v['sha256'] for k, v in manifests.items()},
                      'input_sha256': sha256(input_path), 'output_sha256': sha256(output_path)}))


def diagnose(args):
    from package_model import verify_package
    run = args.run.resolve()
    historical = json.loads((run/'result.json').read_text())
    command = historical['command']
    reference = (run/'oracle/reference').resolve()
    # Saved references may be a direct symlink for non-PE pipeline runs.
    if not reference.is_dir():
        reference = (run/'reference').resolve()
    fixture_path = reference/'fixture.json'
    if sha256(fixture_path) != historical['reference_fixture_sha256']:
        raise ValueError('Historical reference manifest checksum differs')
    fixture = json.loads(fixture_path.read_text())
    if not fixture.get('complete'):
        raise ValueError('Saved official reference is incomplete')
    package = args.model.resolve() if args.model else Path(command[command.index('--model')+1])
    manifest, _ = verify_package(package)
    if manifest['config'] != fixture['config']:
        raise ValueError('VAE package and reference dimensions differ')
    dims = fixture['final']['unpacked']['shape']
    if len(dims) != 4 or dims[:2] != [1, 32]:
        raise ValueError('Expected unpacked NCHW VAE input')
    height, width = dims[2:]
    expected_shape = (1, 3, height*8, width*8)
    if tuple(fixture['final']['decoded']['shape']) != expected_shape:
        raise ValueError('Decoded reference shape differs')
    official_entry = fixture['final']['unpacked']
    if Path(official_entry['file']).name != official_entry['file']:
        raise ValueError('Unsafe reference tensor path')
    official_input = reference/official_entry['file']
    native_input = run/'trace/unpacked.f32'
    native_checks = {row['tensor']: row['sha256'] for row in historical['comparisons']}
    load_tensor(official_input, dims, official_entry['sha256'])
    load_tensor(native_input, dims, native_checks['unpacked'])
    old_official = load_tensor(reference/'decoded.f32', expected_shape,
                               fixture['final']['decoded']['sha256'])
    old_native = load_tensor(run/'trace/decoded.f32', expected_shape, native_checks['decoded'])
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    snapshots = output/'snapshots'
    snapshots.mkdir()
    runner = snapshots/'ernie-head-runner'
    shutil.copy2(args.runner, runner)
    for name in ('diagnose_vae_cross.py', 'export_vae.py', 'package_model.py'):
        shutil.copy2(ROOT/'tools'/name, snapshots/name)
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in
               [ROOT/'sources.lock.json', ROOT/'src/vae.cpp', ROOT/'src/ernie_groupnorm.cpp',
                ROOT/'probes/head_runner.cpp', ROOT/'tools/export_vae.py', Path(__file__)]}
    report = {'scope': 'Four independent CPU FP32 decodes of two saved unpacked latents; diagnostic only',
              'status': 'running', 'historical_pipeline_passed': historical['passed'],
              'historical_result_sha256': sha256(run/'result.json'),
              'reference_manifest_sha256': sha256(fixture_path),
              'package_manifest_sha256': sha256(package/'manifest.json'),
              'native_runner_sha256': sha256(runner), 'sources': sources,
              'native_convolution': 'direct', 'shape': list(expected_shape),
              'pipeline_gates': PIPELINE_FP32, 'standalone_vae_gates': VAE_FP32,
              'runs': {}}
    def save():
        (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    arrays = {}
    try:
        for name, native_decoder, input_path in (
                ('oo', False, official_input), ('no', True, official_input),
                ('on', False, native_input), ('nn', True, native_input)):
            inputs = output/(name+'-input')
            inputs.mkdir()
            shutil.copy2(input_path, inputs/'in0.f32')
            input_hash = sha256(inputs/'in0.f32')
            work = output/name
            if native_decoder:
                command = [str(runner), '--model', str(package/'vae'), '--fixture', str(inputs),
                           '--output', str(work/'actual'), '--component', 'vae',
                           '--width', str(width), '--height', str(height), '--text-tokens', '0',
                           '--backend', 'cpu', '--precision', 'fp32', '--vae-convolution', 'direct']
                tensor = work/'actual/out0.f32'
            else:
                tensor = work/'decoded.f32'
                command = [sys.executable, str(Path(__file__).resolve()), '--official-worker',
                           '--input', str(inputs/'in0.f32'), '--output', str(tensor),
                           '--height', str(height), '--width', str(width)]
            report['runs'][name] = run_child(command, work, args.timeout)
            report['runs'][name].update(input_sha256=input_hash, output_sha256=sha256(tensor))
            arrays[name] = load_tensor(tensor, expected_shape)
            print(json.dumps({'cross_completed': name, **report['runs'][name]}), flush=True)
            save()
        oo, no, on, nn = (arrays[name] for name in ('oo', 'no', 'on', 'nn'))
        report['cross_terms'] = cross_terms(oo, no, on, nn)
        report['comparisons'] = {
            'decoder_on_official': compare_arrays(oo, no, VAE_FP32),
            'decoder_on_native': compare_arrays(on, nn, VAE_FP32),
            'input_through_official': compare_arrays(oo, on, PIPELINE_FP32),
            'connected': compare_arrays(oo, nn, PIPELINE_FP32)}
        report['historical_reproduction'] = {
            'official_bitwise_equal': bool(np.array_equal(old_official, oo)),
            'native_bitwise_equal': bool(np.array_equal(old_native, nn)),
            'official': compare_arrays(old_official, oo, VAE_FP32),
            'native': compare_arrays(old_native, nn, VAE_FP32)}
        report['status'] = 'diagnostic_completed'
        save()
        print(json.dumps({'status': report['status'], 'cross_terms': report['cross_terms'],
                          'comparisons': report['comparisons']}), flush=True)
    except Exception as error:
        report.update(status='failed', failure=str(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--runner', type=Path, default=ROOT/'build/ernie-head-runner')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--official-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--input', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--height', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--width', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists() or args.timeout < 1:
        parser.error('Use a new output and positive timeout')
    if args.official_worker:
        if not args.input or not args.height or not args.width:
            parser.error('Worker requires input and latent dimensions')
        official_decode(args.input, args.output, args.height, args.width)
    else:
        if not args.run:
            parser.error('Require --run with a sealed historical pipeline result')
        diagnose(args)


if __name__ == '__main__':
    main()
