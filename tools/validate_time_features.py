#!/usr/bin/env python3
"""Compare native time features with Diffusers, including fractional timesteps."""
import argparse
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.embeddings import get_timestep_embedding
from export_dit_block import metrics, save_tensor
from prepare_block import ROOT, sha256

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runner', type=Path, default=ROOT / 'build/ernie-denoise-runner')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    runner = args.output / 'runner.snapshot'
    shutil.copy2(args.runner, runner)
    torch.set_num_threads(2)
    values = [0., 1e-8, .25, 1., 62.5, 123.456, 500., 999.99, 1000.]
    values += [float(v) for v in np.random.default_rng(20260905).uniform(0, 1000, 24).astype('f4')]
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=4.)
    scheduler.set_timesteps(sigmas=torch.linspace(1., 0., 9)[:-1], device='cpu')
    values += scheduler.timesteps.tolist()
    manifest = {'scope': 'Time features only; C++ scalar libm versus pinned official CPU FP32',
        'gates': {'max_abs_error': .00013, 'nrmse': .00002}, 'cases': [],
        'generator_sha256': sha256(__file__), 'runner_sha256': sha256(runner),
        'official_source_sha256': sha256(inspect.getfile(get_timestep_embedding))}
    manifest['turbo_timesteps'] = scheduler.timesteps.tolist()
    for i, value in enumerate(values):
        step = torch.tensor([value], dtype=torch.float32)
        expected = get_timestep_embedding(step, 4096, flip_sin_to_cos=False, downscale_freq_shift=0)
        manifest['cases'].append({'timestep': step.item(), 'expected': save_tensor(args.output / f'expected-{i}.f32', expected)})
    path = args.output / 'result.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    for i, case in enumerate(manifest['cases']):
        out = args.output / f'case-{i}'
        subprocess.run([str(runner.resolve()), '--timestep', str(case['timestep']), '--output', str(out.resolve())], check=True, timeout=15)
        actual = torch.from_numpy(np.fromfile(out / 'features.f32', '<f4'))
        expected = torch.from_numpy(np.fromfile(args.output / case['expected']['file'], '<f4'))
        error = metrics(expected, actual)
        case.update(**error, passed=all(error[k] <= limit for k,limit in manifest['gates'].items()), sha256=sha256(out / 'features.f32'))
    invalid = []
    for i, value in enumerate(['-1', '1000.1', 'nan', 'inf', '1garbage']):
        out = args.output / f'invalid-{i}'
        run = subprocess.run([str(runner.resolve()), '--timestep', value, '--output', str(out.resolve())], capture_output=True, text=True, timeout=15)
        invalid.append({'input': value, 'return_code': run.returncode, 'stderr': run.stderr.strip(), 'passed': run.returncode != 0 and not out.exists()})
    manifest['invalid'] = invalid
    manifest['passed'] = all(case['passed'] for case in [*manifest['cases'], *invalid])
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'passed': manifest['passed'], 'valid_cases': len(values), 'invalid_cases': len(invalid),
        'max_abs': max(c['max_abs_error'] for c in manifest['cases']), 'max_nrmse': max(c['nrmse'] for c in manifest['cases'])}))
    return 0 if manifest['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
