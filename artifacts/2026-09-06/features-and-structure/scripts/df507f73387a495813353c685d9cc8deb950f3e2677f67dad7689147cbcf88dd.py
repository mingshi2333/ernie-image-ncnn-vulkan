#!/usr/bin/env python3
"""Pack all 26 pinned PE blocks losslessly and seal an optional native CPU package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import numpy as np
from build_text_weights import BINDINGS, component
from export_pe_block import config
from prepare_block import ROOT, sha256

GRAPH_SHA256 = 'aa068ca4377987878f758359289907a86a3a592132fa6f6b3dadce79f5f1f265'
TEMPLATE_SHA256 = '0c859484eecf01db103acd02c332610163ee425cd46866d6cb126ee1bee974ea'
PE_CONFIG = dict(layers=26, hidden_size=3072, vocabulary=131072, capacity=4096, tokens_per_call=1)


def pack(path, prefix, tensors, target):
    source, header, offset = component(path, prefix)
    if set(header)-{'__metadata__'} != {prefix+name for name, _, _ in tensors}:
        raise ValueError('PE source tensor inventory differs')
    restored = hashlib.sha256()
    with path.open('rb') as file:
        for name, shape, affine in tensors:
            item = header[prefix+name]
            begin, end = item['data_offsets']
            if item['dtype'] != 'BF16' or item['shape'] != shape or end-begin != int(np.prod(shape))*2:
                raise ValueError('PE weight layout differs')
            file.seek(offset+begin)
            if not affine:
                target.write(struct.pack('<I', 0x01348b83)); restored.update(b'\0'*4)
            remaining = end-begin
            while remaining:
                data = file.read(min(remaining, 1024*1024))
                if not data or len(data)%2: raise ValueError('Truncated PE component')
                fp32 = (np.frombuffer(data, '<u2').astype('<u4') << 16).tobytes()
                if not np.isfinite(np.frombuffer(fp32, '<f4')).all(): raise ValueError('Non-finite PE weights')
                target.write(fp32 if affine else data); restored.update(fp32)
                remaining -= len(data)
            if not affine: target.write(b'\0'*((-(end-begin))%4))
    return source, restored.hexdigest()


def runtime_files():
    return sorted(['pe.cfg', 'embeddings.bf16', 'rope-inv-freq.f32', 'head.ncnn.param', 'head.ncnn.bin',
                   'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json', 'tokenizer/chat_template.jinja'] +
                  [f'block-{i:02d}/pe.ncnn.{suffix}' for i in range(26) for suffix in ('param', 'bin')])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True, help='Passing validate_pe_block.py evidence')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Use a new output directory')
    config()
    manifest = json.loads((args.template/'model.json').read_text())
    validation = json.loads((args.validation/'result.json').read_text())
    if (manifest['block'] != 0 or sha256(args.template/'pe.ncnn.param') != GRAPH_SHA256
            or not validation['passed'] or validation['return_code'] != 0
            or validation['model_manifest_sha256'] != sha256(args.template/'model.json')
            or validation['actual_sha256'] != sha256(args.validation/'actual.f32')):
        raise ValueError('Require a verified first PE block and passing native cache validation')
    for name, digest in manifest['files'].items():
        if Path(name).name != name or sha256(args.template/name) != digest:
            raise ValueError('PE template checksum differs')
    tok = ROOT/'models/pe-tokenizer'
    if sha256(tok/'chat_template.jinja') != TEMPLATE_SHA256:
        raise ValueError('PE chat template differs from native formatting')
    for i in range(26):
        if not (ROOT/f'models/official/pe-block-{i:02d}.safetensors').is_file():
            raise ValueError(f'Missing PE block {i}; finish fetch_pe_components.py first')
    # The official config ties the LM head; verify the stored tensors really
    # agree before using a tied FP32 reference in later end-to-end validation.
    embed_meta = json.loads((ROOT/'models/official/pe-embed.manifest.json').read_text())
    head_meta = json.loads((ROOT/'models/official/pe-lm-head.manifest.json').read_text())
    if embed_meta['tensors'][0]['sha256'] != head_meta['tensors'][0]['sha256']:
        raise ValueError('Official PE tied embedding and LM head differ')
    out = args.output.resolve(); out.mkdir(parents=True)
    graph = (args.template/'pe.ncnn.param').read_text()
    bindings = [(BINDINGS[line.split()[1]][0], BINDINGS[line.split()[1]][1], line.split()[0] == 'RMSNorm')
                for line in graph.splitlines()[2:] if line.split()[0] in ('Gemm', 'RMSNorm')]
    block_records = []
    for i in range(26):
        directory = out/f'block-{i:02d}'; directory.mkdir()
        with (directory/'pe.ncnn.bin').open('xb') as target:
            source, restored = pack(ROOT/f'models/official/pe-block-{i:02d}.safetensors', f'model.layers.{i}.', bindings, target)
        if i == 0 and restored != sha256(args.template/'pe.ncnn.bin'):
            raise ValueError('PE BF16 stream does not reconstruct independent FP32 export')
        (directory/'pe.ncnn.param').write_text(graph)
        block_records.append(dict(block=i, weights_sha256=source['sha256'], reconstructed_fp32_sha256=restored))
        print(json.dumps({'packed_pe_block': i}), flush=True)
    path = ROOT/'models/official/pe-embed.safetensors'
    _, header, offset = component(path, 'model.embed_tokens.')
    item = header['model.embed_tokens.weight']; begin, end = item['data_offsets']
    if item['shape'] != [131072, 3072] or item['dtype'] != 'BF16' or end-begin != 131072*3072*2:
        raise ValueError('PE embedding table differs')
    with path.open('rb') as file, (out/'embeddings.bf16').open('xb') as target:
        file.seek(offset+begin)
        remaining = end-begin
        while remaining:
            data = file.read(min(remaining, 1024*1024))
            if not data: raise ValueError('Truncated PE embedding')
            target.write(data); remaining -= len(data)
    (out/'head.ncnn.param').write_text('7767517\n3 3\nInput input 0 1 in0\n'
        'RMSNorm norm 1 1 in0 normalized 0=3072 1=1.000000e-5 2=1\n'
        'Gemm projection 1 1 normalized out0 10=-1 2=0 3=1 4=0 5=1 6=1 7=1 8=131072 9=3072\n')
    with (out/'head.ncnn.bin').open('xb') as target:
        pack(ROOT/'models/official/pe-norm.safetensors', 'model.norm.', [('weight', [3072], True)], target)
        pack(ROOT/'models/official/pe-lm-head.safetensors', 'lm_head.', [('weight', [131072, 3072], False)], target)
    shutil.copyfile(args.template/'rope-inv-freq.f32', out/'rope-inv-freq.f32')
    (out/'tokenizer').mkdir()
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        metadata = json.loads((tok/(name+'.source.json')).read_text())
        if sha256(tok/name) != metadata['sha256']: raise ValueError('PE tokenizer source differs')
        shutil.copyfile(tok/name, out/'tokenizer'/name)
    (out/'pe.cfg').write_text(''.join(f'{k} {v}\n' for k, v in PE_CONFIG.items()))
    lock = json.loads((ROOT/'sources.lock.json').read_text())
    files = runtime_files()
    packed = dict(schema_version=1, kind='prompt_enhancer', portable=True, config=PE_CONFIG,
                  official_model_revision=lock['official_model']['revision'], ncnn_revision=lock['ncnn']['revision'],
                  source_blocks=block_records, builder_sha256=sha256(__file__),
                  template_manifest_sha256=sha256(args.template/'model.json'),
                  cache_validation_sha256=sha256(args.validation/'result.json'),
                  files={name: sha256(out/name) for name in files}, file_sizes={name: (out/name).stat().st_size for name in files})
    (out/'manifest.json').write_text(json.dumps(packed, indent=2)+'\n')
    print(json.dumps({'package': str(out), 'runtime_files': len(files)}), flush=True)


if __name__ == '__main__': main()
