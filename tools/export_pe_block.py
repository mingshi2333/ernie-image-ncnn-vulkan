#!/usr/bin/env python3
"""Export one-token Ministral3 PE blocks with ncnn's native opaque KV cache.

The supported session length is 4096, below the official 16384-position query
scaling boundary. Prefill consumes exact tokens sequentially, without padding.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import torch
from safetensors.torch import load_file
from transformers import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3DecoderLayer, Ministral3RotaryEmbedding
from export_text_block import TextBlock
from export_dit_block import metrics, save_tensor
from prepare_block import ROOT, sha256


def config():
    cfg = Ministral3Config.from_dict(json.loads((ROOT/'models/official/pe-config.json').read_text()))
    required = dict(model_type='ministral3', hidden_size=3072, head_dim=128, num_attention_heads=32,
                    num_key_value_heads=8, num_hidden_layers=26, intermediate_size=9216,
                    rms_norm_eps=1e-5, vocab_size=131072, tie_word_embeddings=True)
    if any(getattr(cfg, k) != v for k, v in required.items()):
        raise ValueError('PE configuration differs from reviewed Ministral3')
    if (cfg.rope_parameters['original_max_position_embeddings'] != 16384
            or cfg.rope_parameters['llama_4_scaling_beta'] != .1):
        raise ValueError('PE query scaling configuration differs')
    cfg._attn_implementation = 'sdpa'
    return cfg


def load_block(index):
    path = ROOT/f'models/official/pe-block-{index:02d}.safetensors'
    manifest = json.loads(path.with_suffix('.manifest.json').read_text())
    prefix = f'model.layers.{index}.'
    revision = json.loads((ROOT/'sources.lock.json').read_text())['official_model']['revision']
    if manifest['prefix'] != prefix or manifest['revision'] != revision or sha256(path) != manifest['sha256']:
        raise ValueError('PE block source differs')
    with torch.device('meta'):
        block = Ministral3DecoderLayer(config(), index)
    state = {k.removeprefix(prefix): v.float() for k, v in load_file(path).items()}
    block.load_state_dict(state, strict=True, assign=True)
    return block.eval().requires_grad_(False), manifest


def cached_graph(graph):
    lines = graph.splitlines()
    layers, blobs = map(int, lines[1].split())
    lines[1] = f'{layers+1} {blobs+4}'
    found = 0
    for i, line in enumerate(lines[2:], 2):
        fields = line.split()
        if fields[0] == 'SDPA':
            if fields[2:4] != ['4', '1'] or fields[9:] != ['5=1']:
                raise ValueError('Unreviewed PE SDPA signature')
            lines[i] = ' '.join(fields[:2]+['6', '3']+fields[4:8]+['past_k', 'past_v', fields[8], 'out_k', 'out_v', '5=1', '7=1'])
            found += 1
    if found != 1:
        raise ValueError('Require one PE attention layer')
    lines.insert(2, 'Input pe_cache 0 2 past_k past_v')
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--block', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or not 0 <= args.block < 26:
        p.error('Use a new output directory and block 0..25')
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    block, source = load_block(args.block)
    wrapper = TextBlock(block).eval()
    rotary = Ministral3RotaryEmbedding(config())
    if rotary.attention_scaling != 1.:
        raise ValueError('Require unit YaRN attention scaling')
    x = torch.randn(1, 17, 3072, generator=torch.Generator().manual_seed(20260906))*.2
    positions = torch.arange(17)
    cos, sin = rotary(x, positions[None])
    mask = torch.full((17, 17), torch.finfo(torch.float32).min).triu(1)
    expected = block(x, position_embeddings=(cos, sin), attention_mask=mask[None, None],
                     cache_position=positions)
    comparison = metrics(expected, wrapper(x, cos, sin, mask))
    if comparison['nrmse'] > 2e-6 or comparison['max_abs_error'] > 2e-5:
        raise ValueError(f'PE wrapper differs from Ministral3: {comparison}')
    out = args.output.resolve(); out.mkdir(parents=True)
    fixture = {'block': args.block, 'tokens': 17, 'capacity': 4096,
               'scope': 'Real PE weights, synthetic 17-token input, official full causal attention versus native incremental cache',
               'official_model_revision': source['revision'], 'weights_sha256': source['sha256'],
               'official_source_sha256': sha256(inspect.getfile(Ministral3DecoderLayer)),
               'config_sha256': sha256(ROOT/'models/official/pe-config.json'),
               'exporter_sha256': sha256(__file__), 'wrapper_vs_official': comparison,
               'inputs': {name: save_tensor(out/(name+'.f32'), value) for name, value in [('x', x), ('cos', cos), ('sin', sin)]},
               'expected': save_tensor(out/'expected.f32', expected),
               'gates': dict(nrmse=.0002, atol=.0002, rtol=.0002)}
    (out/'fixture.json').write_text(json.dumps(fixture, indent=2)+'\n')
    inputs = (x[:, :1].contiguous(), cos[:, :1].contiguous(), sin[:, :1].contiguous(), torch.zeros(1, 1))
    traced = torch.jit.trace(wrapper, inputs, check_trace=False)
    traced.save(str(out/'pe.pt'))
    import pnnx
    binary = Path(pnnx.EXEC_PATH)
    command = [str(binary), 'pe.pt', 'inputshape=[1,1,3072],[1,1,128],[1,1,128],[1,1]', 'fp16=0', 'device=cpu']
    with (out/'conversion.log').open('w') as log:
        rc = subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, 'OMP_NUM_THREADS': '4'}).returncode
    diagnostics = [line for line in (out/'conversion.log').read_text().splitlines()
                   if 'unsupported' in line.lower() or 'not supported' in line]
    graph = (out/'pe.ncnn.param').read_text()
    if rc or diagnostics or any(line.startswith(('aten::', 'pnnx.', 'Tensor.')) for line in graph.splitlines()):
        raise ValueError('PE conversion failed; inspect conversion.log')
    (out/'pe.uncached.ncnn.param').write_text(graph)
    (out/'pe.ncnn.param').write_text(cached_graph(graph))
    rotary.inv_freq.numpy().astype('<f4').tofile(out/'rope-inv-freq.f32')
    conversion = dict(command=command, return_code=rc, unsupported=diagnostics, pnnx_sha256=sha256(binary))
    (out/'conversion.json').write_text(json.dumps(conversion, indent=2)+'\n')
    names = ['pe.ncnn.param', 'pe.uncached.ncnn.param', 'pe.ncnn.bin', 'fixture.json', 'conversion.json', 'rope-inv-freq.f32']
    (out/'model.json').write_text(json.dumps({'block': args.block, 'tokens_per_call': 1, 'capacity': 4096,
        'weights_sha256': source['sha256'], 'files': {name: sha256(out/name) for name in names}}, indent=2)+'\n')
    print(json.dumps({'converted': True, 'block': args.block, 'wrapper_vs_official': comparison}), flush=True)


if __name__ == '__main__':
    main()
