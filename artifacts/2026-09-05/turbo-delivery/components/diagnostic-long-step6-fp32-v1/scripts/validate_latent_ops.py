#!/usr/bin/env python3
"""Compare C++ CPU/Vulkan Euler and BN/unpatchify against the pinned official pipeline."""
import argparse
import inspect
import json
from pathlib import Path
import subprocess
import urllib.request

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ernie_image.pipeline_ernie_image import ErnieImagePipeline
from prepare_block import ROOT, sha256


def save(directory, name, tensor):
    path = directory / name
    value = tensor.detach().cpu().float().contiguous().numpy().astype('<f4')
    value.tofile(path)
    return {'file': name, 'shape': list(value.shape), 'sha256': sha256(path)}


def fixture(directory, config, width, height, steps, lock):
    directory.mkdir(parents=True)
    generator = torch.Generator().manual_seed(20260905 + width * height + steps)
    initial = torch.randn(1, 128, height, width, generator=generator)
    mean = torch.randn(128, generator=generator) * .4
    variance = torch.rand(128, generator=generator) * 2
    variance[:4] = torch.tensor([0., 1e-10, 1e-6, 1e-4])
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
    scheduler.set_timesteps(sigmas=torch.linspace(1., 0., steps + 1)[:-1], device='cpu')
    entries = {name: save(directory, f'{name}.f32', value) for name, value in
               [('initial', initial), ('mean', mean), ('variance', variance)]}
    sample = initial
    for i, timestep in enumerate(scheduler.timesteps):
        prediction = torch.randn(initial.shape, generator=generator) * .3
        entries[f'prediction-{i}'] = save(directory, f'prediction-{i}.f32', prediction)
        sample = scheduler.step(prediction, timestep, sample).prev_sample
        entries[f'step-{i}'] = save(directory, f'expected-step-{i}.f32', sample)
    # This is the actual pinned pipeline's BN epsilon and operation order.
    denormalized = sample * torch.sqrt(variance.reshape(1, 128, 1, 1) + 1e-5) + mean.reshape(1, 128, 1, 1)
    unpacked = ErnieImagePipeline._unpatchify_latents(denormalized)
    if not torch.equal(ErnieImagePipeline._patchify_latents(unpacked), denormalized):
        raise ValueError('Official latent round-trip differs')
    entries['unpacked'] = save(directory, 'expected-unpacked.f32', unpacked)
    result = {'schema_version': 1, 'width': width, 'height': height, 'steps': steps,
              'scope': 'FP32 official scheduler and pipeline transforms; synthetic predictions and BN statistics; no DiT/VAE inference',
              'official_model_revision': lock['official_model']['revision'],
              'diffusers_revision': lock['diffusers']['revision'], 'torch': torch.__version__,
              'sigmas': scheduler.sigmas.tolist(), 'timesteps': scheduler.timesteps.tolist(),
              'gates': {'atol': 2e-6, 'rtol': 2e-6, 'nrmse': 1e-6, 'schedule': 'bit_equal_fp32'},
              'files': entries, 'generator_sha256': sha256(__file__),
              'pipeline_source_sha256': sha256(inspect.getfile(ErnieImagePipeline)),
              'scheduler_source_sha256': sha256(inspect.getfile(FlowMatchEulerDiscreteScheduler))}
    (directory / 'fixture.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def validate(directory, output, runner, data, backend, batched):
    for item in data['files'].values():
        if sha256(directory / item['file']) != item['sha256']:
            raise ValueError('Fixture checksum mismatch')
    command = [str(runner.resolve()), '--fixture', str(directory.resolve()), '--output', str(output.resolve()),
               '--width', str(data['width']), '--height', str(data['height']), '--steps', str(data['steps']),
               '--backend', backend]
    if batched:
        command.append('--batched')
    result = {'command': command, 'runner_sha256': sha256(runner), 'fixture_sha256': sha256(directory / 'fixture.json'),
              'backend': backend, 'batched': batched, 'passed': False}
    try:
        run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=90)
        # A failed runner can exit before creating the directory.
        output.mkdir(parents=True, exist_ok=True)
        (output / 'runner.log').write_text(run.stdout)
        result['return_code'] = run.returncode
        if run.returncode:
            raise ValueError('Runner failed; see runner.log')
        runtime = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')][-1]
        result['runtime'] = runtime
        schedule_exact = all(np.array_equal(np.asarray(runtime[name], np.float32),
                                             np.asarray(data[name], np.float32)) for name in ['sigmas', 'timesteps'])
        result['schedule_exact'] = schedule_exact
        comparisons = []
        for name in [*(f'step-{i}' for i in range(data['steps'])), 'unpacked']:
            reference = np.fromfile(directory / data['files'][name]['file'], '<f4').astype(np.float64)
            actual = np.fromfile(output / f'{name}.f32', '<f4').astype(np.float64)
            if reference.shape != actual.shape or not np.isfinite(actual).all():
                raise ValueError('Invalid output shape or non-finite tensor')
            error = np.abs(actual - reference)
            nrmse = float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-30))
            gates = data['gates']
            passed = np.all(error <= gates['atol'] + gates['rtol'] * np.abs(reference)) and nrmse <= gates['nrmse']
            comparisons.append({'tensor': name, 'max_abs_error': float(error.max()), 'nrmse': nrmse,
                                'passed': bool(passed), 'sha256': sha256(output / f'{name}.f32')})
        result['comparisons'] = comparisons
        result['passed'] = schedule_exact and all(item['passed'] for item in comparisons)
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as error:
        result['failure'] = str(error)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runner', type=Path, default=ROOT / 'build/ernie-latent-runner')
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory to retain previous attempts')
    args.output.mkdir(parents=True)
    lock = json.loads((ROOT / 'sources.lock.json').read_text())
    source = lock['official_model']
    url = f"{source['url']}/resolve/{source['revision']}/scheduler/scheduler_config.json"
    with urllib.request.urlopen(url, timeout=30) as response:
        config_bytes = response.read()
    (args.output / 'scheduler_config.json').write_bytes(config_bytes)
    config = json.loads(config_bytes)
    required = {'shift': 4.0, 'num_train_timesteps': 1000, 'use_dynamic_shifting': False,
                'stochastic_sampling': False, 'invert_sigmas': False, 'shift_terminal': None,
                'use_karras_sigmas': False, 'use_exponential_sigmas': False, 'use_beta_sigmas': False}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError('Official scheduler config differs from the implemented contract')
    # Gates and all reference fixtures exist before any candidate executes.
    cases = [(1, 1, 8), (3, 5, 8), (64, 64, 8), (2, 3, 1), (2, 3, 7), (2, 3, 17), (2, 3, 50)]
    fixtures = []
    for width, height, steps in cases:
        directory = args.output / f'w{width}-h{height}-n{steps}' / 'fixture'
        fixtures.append((directory, fixture(directory, config, width, height, steps, lock)))
    matrix = {'scope': 'FP32 latent operations with saved synthetic predictions, not full denoising inference',
              'scheduler_config_url': url, 'scheduler_config_sha256': sha256(args.output / 'scheduler_config.json'),
              'complete': False, 'passed': False, 'cases': []}
    for directory, data in fixtures:
        variants = [('cpu', False)] if args.cpu_only else [('cpu', False), ('vulkan', False), ('vulkan', True)]
        pair = []
        for backend, batched in variants:
            name = backend + ('-batched' if batched else '')
            output = directory.parent / name
            result = validate(directory, output, args.runner, data, backend, batched)
            row = {'case': directory.parent.name, 'variant': name, 'passed': result['passed'],
                   'result': str((output / 'result.json').relative_to(args.output))}
            matrix['cases'].append(row)
            pair.append(result)
            print(json.dumps(row), flush=True)
            (args.output / 'matrix.json').write_text(json.dumps(matrix, indent=2) + '\n')
        if len(pair) == 3 and pair[1]['passed'] and pair[2]['passed']:
            equal = all(a['sha256'] == b['sha256'] for a, b in zip(pair[1]['comparisons'], pair[2]['comparisons']))
            matrix['cases'][-1]['batched_matches_per_step_exactly'] = equal
            matrix['cases'][-1]['passed'] &= equal
    matrix['complete'] = True
    matrix['passed'] = all(row['passed'] for row in matrix['cases'])
    (args.output / 'matrix.json').write_text(json.dumps(matrix, indent=2) + '\n')
    return 0 if matrix['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
