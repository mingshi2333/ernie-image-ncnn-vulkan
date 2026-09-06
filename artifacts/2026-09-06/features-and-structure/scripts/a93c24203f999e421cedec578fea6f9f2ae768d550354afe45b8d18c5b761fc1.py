#!/usr/bin/env python3
"""Validate generated time features and a streamed DiT/Euler loop against pinned FP32 modules."""
import argparse
import inspect
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from export_dit_block import load_block, make_inputs, metrics, save_tensor
from export_dit_heads import load_heads
from prepare_block import ROOT, sha256
from validate_dit_block import verify as verify_block
from validate_dit_heads import verify as verify_head


def reference(models, pre_path, post_path, output, steps):
    manifests = [verify_block(path, path)[0] for path in models]
    pre, post = verify_head(pre_path), verify_head(post_path)
    if ([m['block'] for m in manifests] != list(range(len(models))) or not 1 <= len(models) <= 36
        or pre['component'] != 'input' or post['component'] != 'output'
        or any(pre[k] != post[k] for k in ('height', 'width', 'tokens', 'text_tokens', 'weights'))
        or any(m['tokens'] != pre['tokens'] for m in manifests)):
        raise ValueError('Require matching heads and a consecutive block prefix')
    model, heads = load_heads()
    if {name: m['sha256'] for name, m in heads.items()} != pre['weights']:
        raise ValueError('Reference and converted head weights differ')
    output.mkdir(parents=True)
    generator = torch.Generator().manual_seed(20260905)
    image = torch.randn(1, 128, pre['height'], pre['width'], generator=generator)
    text = torch.randn(1, pre['text_tokens'], 3072, generator=generator)
    valid = pre['text_tokens'] - 2
    if valid < 0:
        raise ValueError('Fixture needs two padding positions')
    synthetic, freqs = make_inputs(pre['height'], pre['width'], pre['text_tokens'], valid, 20260905)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=4.)
    scheduler.set_timesteps(sigmas=torch.linspace(1., 0., steps + 1)[:-1], device='cpu')
    entries = {'in0': save_tensor(output / 'in0.f32', image), 'in1': save_tensor(output / 'in1.f32', text)}
    for i in range(3):
        entries[f'in{i+3}'] = save_tensor(output / f'in{i+3}.f32', synthetic[7+i])
    fixture = {'scope': 'FP32 official DiT and Euler; saved synthetic text embeddings and latent; no text encoder or VAE',
        'height': pre['height'], 'width': pre['width'], 'text_tokens': pre['text_tokens'], 'tokens': pre['tokens'],
        'valid_text': valid, 'blocks': [m['block'] for m in manifests], 'steps': steps,
        'inputs': entries, 'sigmas': scheduler.sigmas.tolist(), 'timesteps': scheduler.timesteps.tolist(),
        'model_manifests': [sha256(p / 'model.json') for p in [pre_path, *models, post_path]],
        'source_sha256': sha256(__file__), 'torch': torch.__version__,
        'scheduler_config': dict(scheduler.config), 'scheduler_source_sha256': sha256(inspect.getfile(type(scheduler))),
        'gates': {'fp32': {'nrmse': .002, 'global_rtol': .002, 'atol': .0002},
                  'fp16': {'nrmse': .1, 'global_rtol': .1, 'atol': .03},
                  'time_features': {'max_abs': .00013, 'nrmse': .00002}}, 'outputs': [], 'complete': False}
    manifest_path = output / 'fixture.json'
    manifest_path.write_text(json.dumps(fixture, indent=2) + '\n')
    for step, timestep in enumerate(scheduler.timesteps):
        start = time.perf_counter()
        # Hooks obtain the official input projection and shared conditioning.
        captured = {}
        hook = model.final_norm.register_forward_pre_hook(lambda module, inputs: captured.update(x=inputs[0], c=inputs[1]))
        ada = model.adaLN_modulation.register_forward_hook(lambda module, inputs, out: captured.update(ada=out))
        model(image, timestep.reshape(1), text, torch.tensor([valid]))
        hook.remove(); ada.remove()
        features = model.time_proj(timestep.reshape(1))
        x = captured['x']
        temb = [a.view(1, 1, 4096) for a in captured['ada'].chunk(6, dim=-1)]
        for m in manifests:
            block, weight_manifest = load_block(ROOT / f"models/official/dit-block-{m['block']:02d}.safetensors", m['block'])
            if weight_manifest['sha256'] != m['weights_sha256']:
                raise ValueError('Reference and converted block weights differ')
            x = block(x, freqs, temb, attention_mask=synthetic[-1][None, None])
            del block
        patches = model.final_linear(model.final_norm(x, captured['c']))[:pre['height'] * pre['width']]
        prediction = patches.transpose(0, 1).reshape(1, pre['height'], pre['width'], 128).permute(0, 3, 1, 2).contiguous()
        image = scheduler.step(prediction, timestep, image).prev_sample
        fixture['outputs'].append({name: save_tensor(output / f'{name}-{step}.f32', value)
                                   for name, value in [('prediction', prediction), ('step', image), ('features', features)]})
        manifest_path.write_text(json.dumps(fixture, indent=2) + '\n')
        print(json.dumps({'reference_step': step, 'elapsed_seconds': time.perf_counter() - start}), flush=True)
    fixture['complete'] = True
    manifest_path.write_text(json.dumps(fixture, indent=2) + '\n')
    return fixture


def validate(models, pre, post, reference_path, fixture, output, runner, backend, precision):
    command = [str(runner.resolve()), '--fixture', str(reference_path.resolve()), '--output', str(output.resolve()),
               '--input-head', str(pre.resolve()), '--output-head', str(post.resolve()), '--steps', str(fixture['steps']),
               '--width', str(fixture['width']), '--height', str(fixture['height']),
               '--text-tokens', str(fixture['text_tokens']), '--backend', backend, '--precision', precision, '--trace']
    for model in models:
        command += ['--model', str(model.resolve())]
    result = {'backend': backend, 'dit_storage': precision, 'scheduler_storage': 'fp32', 'passed': False,
              'command': command, 'runner_sha256': sha256(runner), 'fixture_sha256': sha256(reference_path / 'fixture.json'),
              'scope': fixture['scope'], 'comparisons': []}
    try:
        with (output.parent / (output.name + '.log')).open('w') as log:
            process = subprocess.Popen(['/usr/bin/time', '-v', *command], stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        result['return_code'] = process.returncode
        if process.returncode:
            raise RuntimeError('Denoiser failed; see log')
        runtime = [json.loads(line) for line in (output.parent / (output.name + '.log')).read_text().splitlines() if line.startswith('{')]
        result['runtime'] = runtime
        if len(runtime) != fixture['steps'] + 1 or not runtime[-1].get('complete'):
            raise RuntimeError('Incomplete step trace')
        if not np.array_equal(np.array([x['timestep'] for x in runtime[:-1]], np.float32), np.array(fixture['timesteps'], np.float32)):
            raise RuntimeError('Timesteps differ')
        for i, entries in enumerate(fixture['outputs']):
            for name, item in entries.items():
                path = reference_path / item['file']
                if sha256(path) != item['sha256']:
                    raise RuntimeError('Reference tensor checksum differs')
                expected = np.fromfile(path, '<f4')
                candidate = output / f'{name}-{i}.f32'
                actual = np.fromfile(candidate, '<f4')
                if expected.shape != actual.shape or not np.isfinite(actual).all():
                    raise RuntimeError('Invalid candidate tensor')
                error = metrics(torch.from_numpy(expected), torch.from_numpy(actual))
                gate = fixture['gates']['time_features' if name == 'features' else precision]
                limit = gate['max_abs'] if name == 'features' else gate['atol'] + gate['global_rtol'] * error['reference_max_abs']
                result['comparisons'].append({'step': i, 'tensor': name, **error,
                    'passed': error['nrmse'] <= gate['nrmse'] and error['max_abs_error'] <= limit, 'sha256': sha256(candidate)})
        if sha256(output / 'final.f32') != sha256(output / f"step-{fixture['steps']-1}.f32"):
            raise RuntimeError('Final latent differs from last Euler step')
        result['passed'] = all(x['passed'] for x in result['comparisons'])
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        result['failure'] = str(error)
    output.mkdir(exist_ok=True)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, action='append', required=True)
    parser.add_argument('--input-head', type=Path, required=True)
    parser.add_argument('--output-head', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--runner', type=Path, default=ROOT / 'build/ernie-denoise-runner')
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 1000:
        parser.error('Use a new output directory and 1..1000 steps')
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    args.output.mkdir(parents=True)
    runner = args.output / 'ernie-denoise-runner.snapshot'
    shutil.copy2(args.runner, runner)
    fixture_path = args.output / 'reference'
    fixture = reference(args.model, args.input_head, args.output_head, fixture_path, args.steps)
    variants = [('cpu', 'fp32')] if args.cpu_only else [('cpu', 'fp32'), ('vulkan', 'fp32'), ('vulkan', 'fp16')]
    results = []
    for backend, precision in variants:
        result = validate(args.model, args.input_head, args.output_head, fixture_path, fixture,
                          args.output / f'{backend}-{precision}', runner, backend, precision)
        results.append(result)
        (args.output / 'matrix.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({k: result.get(k) for k in ('backend', 'dit_storage', 'passed', 'failure')}), flush=True)
    return 0 if all(x['passed'] for x in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
