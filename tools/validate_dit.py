#!/usr/bin/env python3
"""Validate a complete DiT prediction, or an explicit prefix, from saved embeddings."""
import argparse
import json
from pathlib import Path
import shutil
import time
import torch
from export_dit_block import load_block, make_inputs, save_tensor
from export_dit_heads import load_heads
from prepare_block import ROOT, sha256
from validate_dit_block import verify as verify_block
from validate_dit_heads import verify as verify_head
from validate_block_sequence import run


def reference(models, input_head, output_head, output, timestep):
    blocks = [verify_block(path, path)[0] for path in models]
    if [item['block'] for item in blocks] != list(range(len(blocks))) or not 1 <= len(blocks) <= 36:
        raise ValueError('Require a consecutive prefix starting at block zero')
    pre = verify_head(input_head)
    post = verify_head(output_head)
    if pre['component'] != 'input' or post['component'] != 'output' or any(
        pre[key] != post[key] for key in ('height', 'width', 'text_tokens', 'tokens', 'weights')):
        raise ValueError('Heads do not match each other')
    if any(item['tokens'] != pre['tokens'] for item in blocks):
        raise ValueError('Block static token count differs from heads')
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    model, manifests = load_heads()
    if {name: item['sha256'] for name, item in manifests.items()} != pre['weights']:
        raise ValueError('Reference head weights differ from conversion')
    generator = torch.Generator().manual_seed(20260905)
    image = torch.randn(1, 128, pre['height'], pre['width'], generator=generator)
    text = torch.randn(1, pre['text_tokens'], 3072, generator=generator)
    step = torch.tensor([timestep], dtype=torch.float32)
    time_features = model.time_proj(step)
    valid_text = pre['text_tokens'] - 2
    if valid_text < 0:
        raise ValueError('Reference fixture requires at least two padded text positions')
    captured = {}
    hook1 = model.final_norm.register_forward_pre_hook(lambda module, inputs: captured.update(x=inputs[0], c=inputs[1]))
    hook2 = model.adaLN_modulation.register_forward_hook(lambda module, inputs, out: captured.update(modulation=out))
    model(image, step, text, torch.tensor([valid_text]))
    hook1.remove()
    hook2.remove()
    synthetic, freqs = make_inputs(pre['height'], pre['width'], pre['text_tokens'], valid_text, 20260905)
    inputs = (image, text, time_features, *synthetic[7:])
    output.mkdir(parents=True)
    fixture = {'tokens': pre['tokens'], 'height': pre['height'], 'width': pre['width'],
               'text_tokens': pre['text_tokens'], 'valid_text': valid_text, 'timestep': timestep,
               'blocks': [item['block'] for item in blocks],
               'scope': 'Full DiT prediction' if len(blocks) == 36 else 'DiT prefix with input/output heads',
               'input_scope': 'Saved synthetic latent and text embeddings; official conditioning and real weights; no text encoder/VAE/Euler',
               'model_manifests': [sha256(path / 'model.json') for path in [input_head, *models, output_head]],
               'source_sha256': sha256(__file__), 'torch': torch.__version__,
               'inputs': {f'in{i}': save_tensor(output / f'in{i}.f32', value) for i, value in enumerate(inputs)},
               'gates': {'fp32': {'atol': .0002, 'rtol': .0002, 'nrmse': .0002},
                         'fp16': {'atol': .03, 'rtol': .03, 'nrmse': .03},
                         'bf16': {'atol': .2, 'rtol': .2, 'nrmse': .12}},
               'reference_stages': []}
    current = captured['x']
    c = captured['c']
    temb = [value.view(1, 1, 4096) for value in captured['modulation'].chunk(6, dim=-1)]
    for manifest in blocks:
        index = manifest['block']
        started = time.perf_counter()
        block, component = load_block(ROOT / f'models/official/dit-block-{index:02d}.safetensors', index)
        if component['sha256'] != manifest['weights_sha256']:
            raise ValueError('Reference block weights differ from converted model')
        current = block(current, freqs, temb, attention_mask=inputs[-1][None, None])
        fixture['reference_stages'].append({'block': index, 'output': save_tensor(output / f'block-{index:02d}.f32', current),
                                            'elapsed_seconds': time.perf_counter() - started})
        (output / 'fixture.json').write_text(json.dumps(fixture, indent=2) + '\n')
        print(json.dumps({'stage': 'reference', 'block': index}), flush=True)
        del block
    # Use the official final normalization and projection after all selected blocks.
    patches = model.final_linear(model.final_norm(current, c))[:pre['height'] * pre['width']]
    expected = patches.transpose(0, 1).reshape(1, pre['height'], pre['width'], 128).permute(0, 3, 1, 2).contiguous()
    fixture['expected'] = save_tensor(output / 'expected.f32', expected)
    (output / 'fixture.json').write_text(json.dumps(fixture, indent=2) + '\n')
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', type=Path, required=True)
    parser.add_argument('--input-head', type=Path, required=True)
    parser.add_argument('--output-head', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timestep', type=float, default=1000.)
    parser.add_argument('--runner', type=Path, default=ROOT / 'build/ernie-block-sequence-runner')
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--isolated-pipeline-cache', action='store_true', help='Retain the old per-Net cache baseline')
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.timestep <= 1000:
        parser.error('Use a new output directory and finite timestep in [0,1000]')
    args.output.mkdir(parents=True)
    runner = args.output / 'ernie-dit-runner.snapshot'
    shutil.copy2(args.runner, runner)
    fixture_dir = args.output / 'reference'
    fixture = reference(args.model, args.input_head, args.output_head, fixture_dir, args.timestep)
    extra = ['--input-head', str(args.input_head.resolve()), '--output-head', str(args.output_head.resolve()),
             '--width', str(fixture['width']), '--height', str(fixture['height']), '--text-tokens', str(fixture['text_tokens'])]
    if args.isolated_pipeline_cache:
        extra.append('--isolated-pipeline-cache')
    variants = [('cpu', 'fp32')] if args.cpu_only else [('cpu', 'fp32'), ('vulkan', 'fp32'), ('vulkan', 'fp16'), ('vulkan', 'bf16')]
    results = []
    for backend, precision in variants:
        result = run(args.model, fixture_dir, fixture, args.output / f'{backend}-{precision}', runner,
                     backend, precision, 'stream', extra_args=extra)
        result['validator_sha256'] = sha256(__file__)
        result['scope'] = fixture['scope']
        result['input_scope'] = fixture['input_scope']
        (args.output / f'{backend}-{precision}' / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        results.append(result)
        (args.output / 'matrix.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({key: result.get(key) for key in ('backend', 'precision', 'passed', 'nrmse', 'failure', 'max_rss_kib')}), flush=True)
    return 0 if all(item['passed'] for item in results) else 1


if __name__ == '__main__':
    main()
