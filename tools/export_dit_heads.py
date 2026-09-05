#!/usr/bin/env python3
"""Fetch and export the pinned DiT input/conditioning and output components."""
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from diffusers.models.transformers.transformer_ernie_image import ErnieImageTransformer2DModel
from export_dit_block import metrics, save_tensor
from fetch_component import fetch_component
from prepare_block import ROOT, sha256

HEADS = ('x_embedder', 'text_proj', 'time_embedding', 'adaLN_modulation', 'final_norm', 'final_linear')


def load_heads(download=False):
    manifests, state = {}, {}
    revision = json.loads((ROOT / 'sources.lock.json').read_text())['official_model']['revision']
    for name in HEADS:
        path = ROOT / f'models/official/dit-{name}.safetensors'
        if download:
            fetch_component(name + '.', path)
        manifest = json.loads(path.with_suffix('.manifest.json').read_text())
        if (manifest['revision'] != revision or manifest['prefix'] != name + '.'
                or manifest['sha256'] != sha256(path)):
            raise ValueError(f'Unverified component {name}')
        state.update({key: value.float() for key, value in load_file(path).items()})
        manifests[name] = manifest
    with torch.device('meta'):
        model = ErnieImageTransformer2DModel(hidden_size=4096, num_attention_heads=32,
                                             num_layers=0, ffn_hidden_size=12288,
                                             text_in_dim=3072, in_channels=128, out_channels=128,
                                             patch_size=1, rope_axes_dim=(32, 48, 48), eps=1e-6)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False), manifests


class InputHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.x_embedder = model.x_embedder
        self.text_proj = model.text_proj
        self.time_embedding = model.time_embedding
        self.adaLN_modulation = model.adaLN_modulation

    def forward(self, image, text, time_features):
        x = torch.cat((self.x_embedder(image), self.text_proj(text)), dim=1)
        c = self.time_embedding(time_features)
        # One vector per modulation, broadcast inside each block.
        modulation = self.adaLN_modulation(c).reshape(1, 6, 4096)
        return x, c, *(modulation[:, i:i + 1, :] for i in range(6))


class OutputHead(nn.Module):
    def __init__(self, model, height, width):
        super().__init__()
        self.norm = model.final_norm
        self.linear = model.final_linear
        self.height, self.width = height, width

    def forward(self, x, c):
        # Preserve batch=1 throughout export. The official API is sequence-first;
        # tracing its batch transposes confuses pnnx's batch-axis inference.
        scale, shift = self.norm.linear(c).chunk(2, dim=-1)
        normalized = F.layer_norm(x, (4096,), None, None, 1e-6)
        normalized = normalized * (1 + scale.reshape(1, 1, 4096)) + shift.reshape(1, 1, 4096)
        patches = self.linear(normalized)[:, :self.height * self.width]
        return patches.transpose(1, 2).reshape(1, 128, self.height, self.width).contiguous()


def export_component(model, inputs, expected, output, metadata):
    output.mkdir(parents=True)
    actual = model(*inputs)
    actual = actual if isinstance(actual, tuple) else (actual,)
    comparisons = [metrics(ref, value) for ref, value in zip(expected, actual)]
    if len(actual) != len(expected) or any(item['max_abs_error'] != 0 for item in comparisons):
        raise ValueError('Export wrapper differs from pinned official component')
    fixture = {**metadata, 'wrapper_vs_official': comparisons,
               'inputs': {f'in{i}': save_tensor(output / f'in{i}.f32', value) for i, value in enumerate(inputs)},
               'expected': {f'out{i}': save_tensor(output / f'out{i}.f32', value) for i, value in enumerate(expected)},
               'gates': {'fp32': {'atol': .0002, 'rtol': .0002, 'nrmse': .00002},
                         'fp16': {'atol': .03, 'rtol': .03, 'nrmse': .02},
                         'bf16': {'atol': .2, 'rtol': .2, 'nrmse': .1}}}
    (output / 'fixture.json').write_text(json.dumps(fixture, indent=2) + '\n')
    scripted = torch.jit.trace(model, inputs, check_trace=False)
    scripted.save(str(output / 'head.pt'))
    del scripted
    import pnnx
    binary = Path(pnnx.EXEC_PATH)
    shapes = ','.join('[' + ','.join(map(str, value.shape)) + ']' for value in inputs)
    command = [str(binary), 'head.pt', 'inputshape=' + shapes, 'fp16=0', 'device=cpu']
    with (output / 'conversion.log').open('w') as log:
        completed = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'OMP_NUM_THREADS': '4'})
    diagnostics = [line for line in (output / 'conversion.log').read_text().splitlines()
                   if 'not supported' in line or 'unsupported' in line.lower()]
    conversion = {'command': command, 'pnnx_sha256': sha256(binary), 'return_code': completed.returncode,
                  'unsupported_diagnostics': diagnostics}
    param_path = output / 'head.ncnn.param'
    if param_path.exists():
        lines = param_path.read_text().splitlines()[2:]
        conversion['operators'] = sorted({line.split()[0] for line in lines if line.strip()})
        conversion['unconverted'] = [line for line in lines if line.startswith(('pnnx.', 'aten::', 'Tensor.'))]
    (output / 'conversion.json').write_text(json.dumps(conversion, indent=2) + '\n')
    print(json.dumps({'component': metadata['component'], **conversion}), flush=True)
    if completed.returncode or diagnostics or conversion.get('unconverted'):
        raise RuntimeError('Head graph conversion failed; inspect conversion.log')
    names = ['head.ncnn.param', 'head.ncnn.bin', 'fixture.json', 'conversion.json',
             *(entry['file'] for entry in fixture['inputs'].values()),
             *(entry['file'] for entry in fixture['expected'].values())]
    (output / 'model.json').write_text(json.dumps({
        'schema_version': 1, 'component': metadata['component'],
        'source_sha256': sha256(__file__), 'weights': metadata['weights'],
        'ncnn_revision': json.loads((ROOT / 'sources.lock.json').read_text())['ncnn']['revision'],
        'files': {name: sha256(output / name) for name in names}}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--height', type=int, default=4)
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--text-tokens', type=int, default=272)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--only', choices=['input', 'output', 'all'], default='all')
    args = parser.parse_args()
    if args.output.exists() or min(args.height, args.width, args.text_tokens) < 1:
        parser.error('Use a new output directory and positive dimensions')
    if args.height * args.width + args.text_tokens > 6144:
        parser.error('Token count exceeds the supported export range')
    args.output = args.output.resolve()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    model, manifests = load_heads(args.download)
    generator = torch.Generator().manual_seed(20260905)
    image = torch.randn(1, 128, args.height, args.width, generator=generator)
    text = torch.randn(1, args.text_tokens, 3072, generator=generator)
    timestep = torch.tensor([1000.0])
    features = model.time_proj(timestep)
    captured = {}
    hook1 = model.final_norm.register_forward_pre_hook(lambda module, inputs: captured.update(x=inputs[0], c=inputs[1]))
    hook2 = model.adaLN_modulation.register_forward_hook(lambda module, inputs, output: captured.update(modulation=output))
    model(image, timestep, text, torch.tensor([args.text_tokens - 2]))
    hook1.remove()
    hook2.remove()
    metadata = {'height': args.height, 'width': args.width, 'text_tokens': args.text_tokens,
                'tokens': args.height * args.width + args.text_tokens, 'seed': 20260905,
                'weights': {name: item['sha256'] for name, item in manifests.items()},
                'official_revision': next(iter(manifests.values()))['revision'],
                'reference_source_sha256': sha256(inspect.getfile(ErnieImageTransformer2DModel)),
                'scope': 'Real projection/conditioning weights and synthetic inputs; no text encoder or denoising loop'}
    pre_expected = (captured['x'].transpose(0, 1), captured['c'],
                    *(value.view(1, 1, 4096) for value in captured['modulation'].chunk(6, dim=-1)))
    if args.only != 'output':
        export_component(InputHead(model), (image, text, features), pre_expected, args.output / 'input',
                         {**metadata, 'component': 'input'})
    # Exercise large activations so a low-precision normalization overflow is visible.
    x = torch.randn(1, metadata['tokens'], 4096, generator=generator) * 512
    c = captured['c']
    official = model.final_linear(model.final_norm(x.transpose(0, 1), c))[:args.height * args.width]
    expected = official.transpose(0, 1).reshape(1, args.height, args.width, 128).permute(0, 3, 1, 2).contiguous()
    if args.only != 'input':
        export_component(OutputHead(model, args.height, args.width), (x, c), (expected,), args.output / 'output',
                         {**metadata, 'component': 'output', 'activation_scale': 512})


if __name__ == '__main__':
    main()
