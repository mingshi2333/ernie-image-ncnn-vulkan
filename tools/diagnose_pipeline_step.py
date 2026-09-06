#!/usr/bin/env python3
"""Teacher-force one DiT step from a complete official pipeline fixture.

This isolates single-prediction error from differences accumulated by earlier
Euler steps. It does not replace the free-running prompt-to-PNG acceptance test.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.embeddings import get_timestep_embedding
from export_dit_block import save_tensor
from package_model import verify_package
from prepare_block import ROOT, sha256
from validate_block_sequence import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True, help='Zero-based step index')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--runner', type=Path, default=ROOT/'build/ernie-block-sequence-runner')
    args = parser.parse_args()
    package, _ = verify_package(args.model)
    saved = json.loads((args.reference/'fixture.json').read_text())
    cfg = package['config']
    if (args.output.exists() or not saved.get('complete') or saved['config'] != cfg
            or not 0 <= args.step < saved['steps']):
        parser.error('Require a new output, a complete matching reference and a valid step')
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    args.output.mkdir(parents=True)
    runner = args.output/'runner.snapshot'
    shutil.copy2(args.runner, runner)
    scripts = args.output/'scripts'
    scripts.mkdir()
    for path in (ROOT/'tools').glob('*.py'):
        shutil.copy2(path, scripts/path.name)
    fixture_dir = args.output/'fixture'
    fixture_dir.mkdir()
    source_entries = {
        'in0': saved['inputs']['initial'] if args.step == 0 else saved['outputs'][args.step-1]['step'],
        'in1': saved['inputs']['padded-text'],
        **{f'in{i+3}': saved['inputs'][f'constant-{i}'] for i in range(3)},
        'expected': saved['outputs'][args.step]['prediction'],
    }
    entries = {}
    for name, entry in source_entries.items():
        source = args.reference/entry['file']
        if (Path(entry['file']).name != entry['file'] or sha256(source) != entry['sha256']
                or source.stat().st_size != int(np.prod(entry['shape']))*4):
            raise ValueError('Reference tensor checksum or shape differs')
        target = fixture_dir/(name+'.f32')
        shutil.copy2(source, target)
        entries[name] = {**entry, 'file': target.name}
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=4.)
    scheduler.set_timesteps(sigmas=torch.linspace(1., 0., saved['steps']+1)[:-1], device='cpu')
    timestep = scheduler.timesteps[args.step].reshape(1)
    features = get_timestep_embedding(timestep, 4096, flip_sin_to_cos=False, downscale_freq_shift=0)
    entries['in2'] = save_tensor(fixture_dir/'in2.f32', features)
    fixture = {
        'scope': 'One teacher-forced DiT prediction; official input latent, text, time features, RoPE and mask',
        'config': cfg, 'step': args.step, 'timestep': timestep.item(),
        'tokens': cfg['packed_height']*cfg['packed_width']+cfg['dit_text_tokens'],
        'inputs': {k: v for k, v in entries.items() if k != 'expected'},
        'expected': entries['expected'],
        'reference_fixture_sha256': sha256(args.reference/'fixture.json'),
        'package_manifest_sha256': sha256(args.model/'manifest.json'),
        'source_snapshot': {p.name: sha256(p) for p in sorted(scripts.iterdir())},
        # The existing full-pipeline tensor gates, unchanged.
        'gates': {'fp32': {'atol': .0002, 'rtol': .01, 'nrmse': .003},
                  'fp16': {'atol': .03, 'rtol': .25, 'nrmse': .15},
                  'bf16': {'atol': .03, 'rtol': .25, 'nrmse': .15}},
    }
    (fixture_dir/'fixture.json').write_text(json.dumps(fixture, indent=2)+'\n')
    extra = ['--input-head', str((args.model/'dit/input').resolve()),
             '--output-head', str((args.model/'dit/output').resolve()),
             '--width', str(cfg['packed_width']), '--height', str(cfg['packed_height']),
             '--text-tokens', str(cfg['dit_text_tokens'])]
    result = run([args.model/f'dit/block-{i:02d}' for i in range(36)], fixture_dir, fixture,
                 args.output/'native', runner, 'vulkan', args.precision, 'stream', extra_args=extra)
    print(json.dumps({k: result.get(k) for k in ('passed', 'failure', 'nrmse', 'max_abs_error', 'max_abs_limit')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
