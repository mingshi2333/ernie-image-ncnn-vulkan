#!/usr/bin/env python3
"""Export the actual Mistral text path selected by the pinned ERNIE config."""
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
from transformers import Mistral3Config
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRotaryEmbedding
from export_dit_block import ExportBlock, metrics, save_tensor
from prepare_block import ROOT, sha256


def config():
    value = Mistral3Config.from_dict(json.loads((ROOT / 'models/official/text_encoder-config.json').read_text())).text_config
    required = {'model_type': 'mistral', 'hidden_size': 3072, 'head_dim': 128,
                'num_attention_heads': 32, 'num_key_value_heads': 8, 'num_hidden_layers': 26,
                'intermediate_size': 9216, 'rms_norm_eps': 1e-5}
    if any(getattr(value, key) != expected for key, expected in required.items()):
        raise ValueError('Text config differs from the reviewed Mistral path')
    value._attn_implementation = 'sdpa'
    value.use_cache = False
    return value


def load_block(index):
    path = ROOT / f'models/official/text-block-{index:02d}.safetensors'
    manifest = json.loads(path.with_suffix('.manifest.json').read_text())
    lock = json.loads((ROOT / 'sources.lock.json').read_text())
    prefix = f'language_model.model.layers.{index}.'
    if (manifest['revision'] != lock['official_model']['revision'] or manifest['prefix'] != prefix
        or sha256(path) != manifest['sha256']):
        raise ValueError('Invalid text component manifest')
    with torch.device('meta'):
        block = MistralDecoderLayer(config(), index)
    state = {name.removeprefix(prefix): value.float() for name, value in load_file(path).items()}
    block.load_state_dict(state, strict=True, assign=True)
    return block.eval().requires_grad_(False), manifest


class TextBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, cos, sin, mask):
        b = self.block
        norm = F.rms_norm(x, (3072,), b.input_layernorm.weight, 1e-5)
        q = b.self_attn.q_proj(norm).reshape(1, -1, 32, 128).transpose(1, 2)
        k = b.self_attn.k_proj(norm).reshape(1, -1, 8, 128).transpose(1, 2)
        v = b.self_attn.v_proj(norm).reshape(1, -1, 8, 128).transpose(1, 2)
        q = ExportBlock.rotary(q, cos, sin)
        k = ExportBlock.rotary(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, mask, dropout_p=0., is_causal=False, enable_gqa=True)
        x = x + b.self_attn.o_proj(a.transpose(1, 2).reshape(1, -1, 4096))
        return x + b.mlp(F.rms_norm(x, (3072,), b.post_attention_layernorm.weight, 1e-5))


def export(index, tokens, output):
    block, weights = load_block(index)
    torch.manual_seed(20260905)
    x = torch.randn(1, tokens, 3072) * .2
    position_ids = torch.arange(tokens)[None]
    rotary = MistralRotaryEmbedding(config())
    cos, sin = rotary(x, position_ids)
    # Full prefill, causal mask; no cross-prompt padding in the text model.
    mask = torch.full((tokens, tokens), torch.finfo(torch.float32).min).triu(1)
    expected = block(x, attention_mask=mask[None, None], position_ids=position_ids,
                     position_embeddings=(cos, sin), use_cache=False)
    wrapper = TextBlock(block).eval()
    inputs = (x, cos, sin, mask)
    comparison = metrics(expected, wrapper(*inputs))
    if comparison['nrmse'] > 2e-6 or comparison['max_abs_error'] > 2e-5:
        raise ValueError(f'Text export wrapper differs: {comparison}')
    output.mkdir(parents=True)
    fixture = {'block': index, 'tokens': tokens, 'scope': 'Official Mistral text block weights, synthetic activation, causal YaRN prefill',
        'weights_sha256': weights['sha256'], 'official_model_revision': weights['revision'],
        'effective_class': 'MistralDecoderLayer', 'llama4_attention_scaling': False,
        'rope_attention_scaling': rotary.attention_scaling, 'wrapper_vs_official': comparison,
        'config_sha256': sha256(ROOT / 'models/official/text_encoder-config.json'),
        'official_source_sha256': sha256(inspect.getfile(MistralDecoderLayer)), 'exporter_sha256': sha256(__file__),
        'inputs': {f'in{i}': save_tensor(output / f'in{i}.f32', v) for i, v in enumerate(inputs)},
        'expected': save_tensor(output / 'out0.f32', expected),
        'gates': {'fp32': {'nrmse': .00002, 'atol': .0002, 'rtol': .0002},
                  'fp16': {'nrmse': .02, 'atol': .03, 'rtol': .03},
                  'bf16': {'nrmse': .1, 'atol': .2, 'rtol': .2}}}
    (output / 'fixture.json').write_text(json.dumps(fixture, indent=2) + '\n')
    traced = torch.jit.trace(wrapper, inputs, check_trace=False)
    traced.save(str(output / 'text.pt'))
    import pnnx
    binary = Path(pnnx.EXEC_PATH)
    command = [str(binary), 'text.pt', 'inputshape=' + ','.join('[' + ','.join(map(str,v.shape)) + ']' for v in inputs), 'fp16=0', 'device=cpu']
    with (output / 'conversion.log').open('w') as log:
        result = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'OMP_NUM_THREADS': '4'})
    diagnostics = [s for s in (output / 'conversion.log').read_text().splitlines() if 'not supported' in s or 'unsupported' in s.lower()]
    graph = (output / 'text.ncnn.param').read_text() if (output / 'text.ncnn.param').is_file() else ''
    conversion = {'return_code': result.returncode, 'unsupported': diagnostics, 'pnnx_sha256': sha256(binary),
                  'operators': sorted({s.split()[0] for s in graph.splitlines()[2:]}),
                  'unconverted': [s for s in graph.splitlines()[2:] if s.startswith(('aten::','pnnx.','Tensor.'))]}
    (output / 'conversion.json').write_text(json.dumps(conversion, indent=2) + '\n')
    print(json.dumps(conversion), flush=True)
    if result.returncode or not graph or diagnostics or conversion['unconverted']:
        raise RuntimeError('Text graph conversion failed')
    (output / 'text.pnnx-original.param').write_text(graph)
    prepared, reshaped = [], 0
    for line in graph.splitlines():
        fields = line.split()
        if fields and fields[0] == 'ExpandDims':
            if fields[2:4] != ['1', '1'] or fields[4] not in ('in1', 'in2') or fields[6:] != ['-23303=1,0']:
                raise ValueError('Unexpected text rotary broadcast')
            fields[0] = 'Reshape'
            line = ' '.join(fields[:6] + ['0=128', f'1={tokens}', '2=1'])
            reshaped += 1
        prepared.append(line)
    if reshaped != 2 or graph.count('\nSDPA ') != 1 or graph.count('\nRMSNorm ') != 2:
        raise ValueError('Unexpected text graph structure')
    (output / 'text.ncnn.param').write_text('\n'.join(prepared) + '\n')
    names = ['text.ncnn.param', 'text.pnnx-original.param', 'text.ncnn.bin', 'fixture.json', 'conversion.json']
    (output / 'model.json').write_text(json.dumps({'block': index, 'tokens': tokens, 'weights_sha256': weights['sha256'],
        'files': {name: sha256(output / name) for name in names}}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.block < 25 or not 1 <= args.tokens <= 2048:
        parser.error('Use a new output, block 0..24, tokens 1..2048')
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    export(args.block, args.tokens, args.output.resolve())

if __name__ == '__main__':
    main()
