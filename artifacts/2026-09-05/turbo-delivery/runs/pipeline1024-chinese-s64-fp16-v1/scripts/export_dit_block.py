#!/usr/bin/env python3
"""Export one real-weight ERNIE DiT block and retain an official-reference fixture."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from diffusers.models.transformers.transformer_ernie_image import ErnieImageSharedAdaLNBlock, ErnieImageEmbedND3
from prepare_block import prepare

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class ExportBlock(nn.Module):
    """Remove attention dispatch plumbing while preserving the pinned block equations."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    @staticmethod
    def rotary(x, cos, sin):
        first = x[..., :64]
        second = x[..., 64:]
        # ERNIE's repeated full-width angles are not equal in the two halves.
        # Express both halves explicitly: pnnx's standard RotaryEmbed fusion
        # assumes one shared half-width cos/sin table and would change the math.
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return torch.cat((first * cos[..., :64] - second * sin[..., :64],
                          second * cos[..., 64:] + first * sin[..., 64:]), dim=-1)

    def forward(self, x, shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp, cos, sin, mask):
        block = self.block
        residual = x
        hidden = F.rms_norm(x, (4096,), block.adaLN_sa_ln.weight, 1e-6)
        hidden = hidden * (1.0 + scale_sa) + shift_sa
        attn = block.self_attention
        q = attn.to_q(hidden).reshape(1, -1, 32, 128).transpose(1, 2)
        k = attn.to_k(hidden).reshape(1, -1, 32, 128).transpose(1, 2)
        v = attn.to_v(hidden).reshape(1, -1, 32, 128).transpose(1, 2)
        q = self.rotary(attn.norm_q(q), cos, sin)
        k = self.rotary(attn.norm_k(k), cos, sin)
        hidden = F.scaled_dot_product_attention(q, k, v, mask,
                                                dropout_p=0.0, is_causal=False)
        hidden = hidden.transpose(1, 2).reshape(1, -1, 4096)
        hidden = attn.to_out[0](hidden)
        x = residual + gate_sa * hidden
        residual = x
        x = F.rms_norm(x, (4096,), block.adaLN_mlp_ln.weight, 1e-6)
        x = x * (1.0 + scale_mlp) + shift_mlp
        return residual + gate_mlp * block.mlp(x)


def load_block(path, index):
    manifest = json.loads(path.with_suffix('.manifest.json').read_text())
    if sha256(path) != manifest['sha256']:
        raise ValueError('Component checksum mismatch')
    with torch.device('meta'):
        block = ErnieImageSharedAdaLNBlock(4096, 32, 12288, eps=1e-6)
    prefix = f'layers.{index}.'
    state = {name.removeprefix(prefix): value.float() for name, value in load_file(path).items()}
    block.load_state_dict(state, strict=True, assign=True)
    return block.eval().requires_grad_(False), manifest


def make_inputs(height, width, text_tokens, valid_text, seed):
    generator = torch.Generator().manual_seed(seed)
    image_tokens = height * width
    length = image_tokens + text_tokens
    hidden = torch.randn(length, 4096, generator=generator)
    temb = [torch.randn(4096, generator=generator) * 0.15 for _ in range(6)]
    temb[2] += 0.5
    temb[5] += 0.5
    grid = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij'), dim=-1).reshape(-1, 2)
    image_ids = torch.cat([torch.full((image_tokens, 1), valid_text), grid], dim=-1)
    text_ids = torch.stack([torch.arange(text_tokens), torch.zeros(text_tokens), torch.zeros(text_tokens)], dim=-1)
    positions = torch.cat([image_ids, text_ids], dim=0).float().unsqueeze(0)
    freqs = ErnieImageEmbedND3(128, 256, (32, 48, 48))(positions)
    cos, sin = freqs[0, :, 0].cos(), freqs[0, :, 0].sin()
    mask = torch.zeros(length, length)
    mask[:, image_tokens + valid_text:] = -1e30
    # Keep batch=1 explicit throughout tracing so pnnx can remove it consistently.
    return (hidden.unsqueeze(0), *(x.view(1, 1, -1) for x in temb), cos.unsqueeze(0), sin.unsqueeze(0), mask), freqs


def metrics(a, b):
    delta = (a.double() - b.double()).flatten()
    ref = a.double().flatten()
    return {'max_abs_error': delta.abs().max().item(), 'nrmse': (delta.square().sum() / ref.square().sum().clamp_min(1e-30)).sqrt().item(),
            'reference_max_abs': ref.abs().max().item()}


def save_tensor(path, value):
    data = value.detach().cpu().float().contiguous().numpy().astype('<f4', copy=False)
    data.tofile(path)
    return {'file': path.name, 'shape': list(data.shape), 'dtype': 'float32_le', 'sha256': sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/converted/dit-block-00')
    parser.add_argument('--height', type=int, default=4, help='Latent patch grid height')
    parser.add_argument('--width', type=int, default=4, help='Latent patch grid width')
    parser.add_argument('--text-tokens', type=int, default=8)
    parser.add_argument('--valid-text', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20260905)
    parser.add_argument('--pnnx', type=Path)
    parser.add_argument('--fixture-only', action='store_true')
    args = parser.parse_args()
    if args.height < 1 or args.width < 1 or not 0 <= args.valid_text <= args.text_tokens:
        parser.error('Invalid grid or text lengths')
    if not 0 <= args.block < 36 or args.height * args.width + args.text_tokens > 6144:
        parser.error('Block index or token count exceeds the supported export range')
    weights = args.weights or ROOT / 'models/official' / f'dit-block-{args.block:02d}.safetensors'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'fixture.json').exists():
        parser.error('Fixture already exists; use a new output path to retain evidence')
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    block, weight_manifest = load_block(weights, args.block)
    model = ExportBlock(block).eval()
    inputs, freqs = make_inputs(args.height, args.width, args.text_tokens, args.valid_text, args.seed)
    temb = [item.view(1, 1, -1) for item in inputs[1:7]]
    start = time.perf_counter()
    expected = block(inputs[0].transpose(0, 1), freqs, temb, attention_mask=inputs[-1][None, None]).transpose(0, 1)
    candidate = model(*inputs)
    comparison = metrics(expected, candidate)
    print(json.dumps({'stage': 'wrapper_vs_official', **comparison}), flush=True)
    if not torch.allclose(expected, candidate, atol=2e-5, rtol=2e-5):
        raise RuntimeError('Export wrapper differs from the official block')
    fixture = {'schema_version': 1, 'block': args.block, 'seed': args.seed, 'tokens': inputs[0].shape[1],
               'grid': [args.height, args.width], 'text_tokens': args.text_tokens, 'valid_text': args.valid_text,
               'weights_sha256': weight_manifest['sha256'], 'official_model_revision': weight_manifest['revision'],
               'input_scope': 'Real block weights; synthetic hidden states and shared AdaLN vectors; official position embeddings',
               'wrapper_vs_official': comparison, 'reference_generation_seconds': time.perf_counter() - start,
               'reference_source_sha256': sha256(inspect.getfile(ErnieImageSharedAdaLNBlock)),
               'versions': {name: importlib.metadata.version(name) for name in ['torch', 'diffusers', 'transformers', 'safetensors', 'pnnx']},
               'torch_build': torch.__version__,
               'inputs': {f'in{i}': save_tensor(out / f'in{i}.f32', value) for i, value in enumerate(inputs)},
               'expected': save_tensor(out / 'expected.f32', expected),
               'gates': {'fp32': {'atol': 0.0002, 'rtol': 0.0002, 'nrmse': 0.00002},
                         'fp16': {'atol': 0.02, 'rtol': 0.02, 'nrmse': 0.01},
                         'bf16': {'atol': 0.15, 'rtol': 0.08, 'nrmse': 0.03}}}
    # Commit numerical gates before any ncnn candidate is run. These gates apply to this fixture family only.
    (out / 'fixture.json').write_text(json.dumps(fixture, indent=2) + '\n')
    if args.fixture_only:
        return
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        scripted = torch.jit.trace(model, inputs, check_trace=False)
    scripted.save(str(out / 'block.pt'))
    del scripted
    if args.pnnx:
        binary = args.pnnx.resolve()
    else:
        import pnnx
        binary = Path(pnnx.EXEC_PATH)
    shapes = ','.join('[' + ','.join(map(str, tensor.shape)) + ']' for tensor in inputs)
    command = [str(binary), 'block.pt', 'inputshape=' + shapes, 'fp16=0', 'device=cpu']
    print(json.dumps({'stage': 'pnnx', 'binary': str(binary)}), flush=True)
    with (out / 'conversion.log').open('w') as log:
        result = subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'OMP_NUM_THREADS': '4'})
    conversion = {'command': command, 'pnnx_sha256': sha256(binary), 'return_code': result.returncode,
                  'exporter_sha256': sha256(__file__),
                  'unsupported_diagnostics': [line for line in (out / 'conversion.log').read_text().splitlines()
                                              if 'not supported' in line or 'unsupported' in line.lower()]}
    if result.returncode == 0 and (out / 'block.ncnn.param').is_file():
        param = (out / 'block.ncnn.param').read_text()
        conversion['sdpa_count'] = sum(line.startswith('SDPA ') for line in param.splitlines())
        conversion['rmsnorm_count'] = sum(line.startswith('RMSNorm ') for line in param.splitlines())
        conversion['unsafe_rotary_fusion'] = any(line.startswith('RotaryEmbed ') for line in param.splitlines())
        conversion['unconverted_operators'] = [line.split()[:2] for line in param.splitlines() if line.startswith(('pnnx.', 'Tensor.', 'aten::'))]
    (out / 'conversion.json').write_text(json.dumps(conversion, indent=2) + '\n')
    print(json.dumps(conversion), flush=True)
    if (result.returncode != 0 or conversion.get('sdpa_count') != 1 or conversion.get('rmsnorm_count') != 4
            or conversion.get('unsafe_rotary_fusion') or conversion.get('unconverted_operators') or conversion['unsupported_diagnostics']):
        raise RuntimeError('Conversion did not produce the required native SDPA block; inspect conversion.log')
    prepare(out, out / 'runtime')
    print(json.dumps({'stage': 'runtime_ready', 'path': str(out / 'runtime')}), flush=True)


if __name__ == '__main__':
    main()
