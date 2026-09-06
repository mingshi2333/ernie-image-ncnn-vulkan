#!/usr/bin/env python3
"""Locate DiT error using identical real pipeline inputs and saved stage outputs.

This is a teacher-forced diagnostic, not free-running quality acceptance. The
official CUDA process exits before Vulkan starts, so they never share VRAM.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from export_dit_block import load_block, make_inputs, metrics, save_tensor
from export_dit_heads import InputHead, load_heads
from prepare_block import ROOT, sha256
from validate_block_sequence import run


def tensor(path, entry):
    source = path / entry['file']
    if (Path(entry['file']).name != entry['file'] or sha256(source) != entry['sha256']
            or source.stat().st_size != int(np.prod(entry['shape'])) * 4):
        raise ValueError('Reference tensor checksum or shape differs')
    return torch.from_numpy(np.fromfile(source, '<f4').copy()).reshape(entry['shape'])


def oracle(args):
    manifest = json.loads((args.model / 'manifest.json').read_text())
    cfg = manifest['config']
    saved = json.loads((args.reference / 'fixture.json').read_text())
    if not saved.get('complete') or cfg != saved['config'] or not 0 <= args.step < saved['steps']:
        raise ValueError('Require a complete matching pipeline reference and valid step')
    h, w, text = cfg['packed_height'], cfg['packed_width'], cfg['dit_text_tokens']
    latent_entry = saved['inputs']['initial'] if args.step == 0 else saved['outputs'][args.step-1]['step']
    latent = tensor(args.reference, latent_entry)
    embeddings = tensor(args.reference, saved['inputs']['padded-text'])
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=4.)
    scheduler.set_timesteps(sigmas=torch.linspace(1., 0., saved['steps']+1)[:-1], device='cpu')
    heads, _ = load_heads()
    time_features = heads.time_proj(scheduler.timesteps[args.step].reshape(1))
    synth, freqs = make_inputs(h, w, text, len(saved['ids']), 20260905)
    constants = [tensor(args.reference, saved['inputs'][f'constant-{i}']) for i in range(3)]
    if any(not torch.equal(a, b) for a, b in zip(constants, synth[7:])):
        raise ValueError('Regenerated official RoPE/mask differ from the saved reference')
    fixture_dir = args.output / 'fixture'
    fixture_dir.mkdir()
    inputs = [latent, embeddings, time_features, *constants]
    fixture = {'scope': 'Teacher-forced real DiT with stage traces; saved latent, text and CPU time features',
               'rope_scope': ('Native uses saved CPU-generated cos/sin tables; the official block evaluates '
                              'cos/sin from the same frequencies on its reference device. Last-bit '
                              'CPU/CUDA trigonometric differences remain part of this comparison.'),
               'config': cfg, 'step': args.step, 'tokens': h*w+text,
               'inputs': {f'in{i}': save_tensor(fixture_dir/f'in{i}.f32', v) for i, v in enumerate(inputs)},
               'reference_fixture_sha256': sha256(args.reference/'fixture.json'),
               'reference_device': args.reference_device, 'source_sha256': sha256(__file__),
               'gates': {'fp32': {'atol': .0002, 'rtol': .01, 'nrmse': .003},
                         'fp16': {'atol': .03, 'rtol': .25, 'nrmse': .15}}, 'stages': {}}
    projected = InputHead(heads)(*inputs[:3])
    for i, value in enumerate(projected):
        fixture['stages'][f'head-{i}'] = save_tensor(fixture_dir/f'head-{i}.f32', value)
    device = args.reference_device
    current = projected[0].transpose(0, 1).to(device)
    temb = [v.to(device) for v in projected[2:]]
    freqs = freqs.to(device)
    mask = constants[-1][None, None].to(device)
    for index in range(36):
        block, component = load_block(ROOT/f'models/official/dit-block-{index:02d}.safetensors', index)
        if component['sha256'] != manifest['source_weights']['dit'][index]:
            raise ValueError('Official weights differ from package')
        current = block.to(device)(current, freqs, temb, attention_mask=mask)
        fixture['stages'][f'block-{index}'] = save_tensor(fixture_dir/f'block-{index}.f32', current.transpose(0, 1))
        del block
        if (index+1) % 6 == 0:
            print(json.dumps({'reference_blocks_complete': index+1}), flush=True)
    patches = heads.final_linear(heads.final_norm(current.cpu(), projected[1]))[:h*w]
    prediction = patches.transpose(0, 1).reshape(1, h, w, 128).permute(0, 3, 1, 2).contiguous()
    fixture['expected'] = save_tensor(fixture_dir/'expected.f32', prediction)
    fixture['saved_prediction_comparison'] = metrics(tensor(args.reference, saved['outputs'][args.step]['prediction']), prediction)
    (fixture_dir/'fixture.json').write_text(json.dumps(fixture, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--reference-device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--oracle-only', action='store_true')
    parser.add_argument('--runner', type=Path, default=ROOT/'build/ernie-block-sequence-runner')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Require a new output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.oracle_only:
        oracle(args)
        return 0
    scripts = args.output/'scripts'
    scripts.mkdir()
    for source in (ROOT/'tools').glob('*.py'):
        shutil.copy2(source, scripts/source.name)
    runner = args.output/'runner.snapshot'
    shutil.copy2(args.runner, runner)
    command = [sys.executable, str(Path(__file__).resolve()), '--model', str(args.model.resolve()),
               '--reference', str(args.reference.resolve()), '--step', str(args.step),
               '--output', str((args.output/'oracle').resolve()), '--reference-device', args.reference_device,
               '--oracle-only']
    with (args.output/'oracle.log').open('w') as log:
        subprocess.run(command, check=True, timeout=1800, stdout=log, stderr=subprocess.STDOUT)
    fixture_dir = args.output/'oracle/fixture'
    fixture = json.loads((fixture_dir/'fixture.json').read_text())
    cfg = fixture['config']
    extra = ['--input-head', str((args.model/'dit/input').resolve()),
             '--output-head', str((args.model/'dit/output').resolve()),
             '--width', str(cfg['packed_width']), '--height', str(cfg['packed_height']),
             '--text-tokens', str(cfg['dit_text_tokens']), '--trace-dir', str((args.output/'trace').resolve())]
    result = run([args.model/f'dit/block-{i:02d}' for i in range(36)], fixture_dir, fixture,
                 args.output/'native', runner, 'vulkan', args.precision, 'stream', extra_args=extra)
    comparisons = []
    if 'failure' not in result:
        for name, entry in fixture['stages'].items():
            expected = tensor(fixture_dir, entry)
            actual = torch.from_numpy(np.fromfile(args.output/'trace'/f'{name}.f32', '<f4').copy()).reshape(expected.shape)
            error = {'stage': name, **metrics(expected, actual), 'actual_sha256': sha256(args.output/'trace'/f'{name}.f32')}
            if expected.numel() == fixture['tokens']*4096:
                n = cfg['packed_height']*cfg['packed_width']
                error['image_tokens'] = metrics(expected[:, :n], actual[:, :n])
                error['text_tokens'] = metrics(expected[:, n:], actual[:, n:])
            comparisons.append(error)
    report = {'scope': fixture['scope'], 'rope_scope': fixture['rope_scope'], 'prediction': result, 'stages': comparisons,
              'saved_prediction_comparison': fixture['saved_prediction_comparison'],
              'source_snapshot': {p.name: sha256(p) for p in sorted(scripts.iterdir())}}
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'prediction': result, 'stages': comparisons}), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
