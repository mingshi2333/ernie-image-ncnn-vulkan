#!/usr/bin/env python3
"""Write official BF16 DiT weights directly into an already reviewed static ncnn graph."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import time
import numpy as np
from prepare_block import ROOT, sha256
from validate_dit_block import verify

# Canonical hash covers the entire reviewed graph, normalizing only token-count
# parameters. All three exported buckets have this hash. A graph change needs a
# new review; matching the number of layers or weight shapes alone is insufficient.
GRAPH_SHA256 = '88206e024759e4f8981fa1cb04636e644c3c7a3c7e08b2658a1f9bcce52ea50c'
BINDINGS = {
    'rmsn_8': ('adaLN_sa_ln.weight', [4096]),
    'gemm_0': ('self_attention.to_q.weight', [4096, 4096]),
    'gemm_1': ('self_attention.to_k.weight', [4096, 4096]),
    'gemm_2': ('self_attention.to_v.weight', [4096, 4096]),
    'rmsn_9': ('self_attention.norm_q.weight', [128]),
    'rmsn_10': ('self_attention.norm_k.weight', [128]),
    'gemm_3': ('self_attention.to_out.0.weight', [4096, 4096]),
    'rmsn_11': ('adaLN_mlp_ln.weight', [4096]),
    'gemm_4': ('mlp.up_proj.weight', [12288, 4096]),
    'gemm_5': ('mlp.gate_proj.weight', [12288, 4096]),
    'gemm_6': ('mlp.linear_fc2.weight', [4096, 12288]),
}


def graph_hash(text, tokens):
    lines = []
    for line in text.splitlines():
        fields = line.split()
        key = None
        if fields and fields[0] == 'Gemm':
            key = '7'
        elif len(fields) > 1 and fields[1] in ['reshape_12', 'reshape_13', 'reshape_14']:
            key = '2'
        elif len(fields) > 1 and fields[1] in ['reshape_15', 'unsqueeze_20', 'unsqueeze_21']:
            key = '1'
        if key is not None:
            if f'{key}={tokens}' not in fields:
                raise ValueError('Static token dimension differs from the model manifest')
            fields = [f'{key}=TOKENS' if item == f'{key}={tokens}' else item for item in fields]
        lines.append(' '.join(fields))
    return hashlib.sha256(('\n'.join(lines) + '\n').encode()).hexdigest()


def build(template, weights, fixture, output):
    template, weights, fixture, output = [Path(p).resolve() for p in (template, weights, fixture, output)]
    started = time.perf_counter()
    template_manifest, template_fixture = verify(template, template)
    graph = (template / 'block.ncnn.param').read_text()
    if graph_hash(graph, template_manifest['tokens']) != GRAPH_SHA256:
        raise ValueError('Graph differs from the reviewed ERNIE topology and weight bindings')
    source = json.loads(weights.with_suffix('.manifest.json').read_text())
    reference = json.loads((fixture / 'fixture.json').read_text())
    lock = json.loads((ROOT / 'sources.lock.json').read_text())
    block = reference['block']
    if not 0 <= block < 36 or source['prefix'] != f'layers.{block}.':
        raise ValueError('Block index and component prefix differ')
    if (source['revision'] != lock['official_model']['revision']
            or reference['official_model_revision'] != source['revision']
            or source['sha256'] != reference['weights_sha256'] or sha256(weights) != source['sha256']):
        raise ValueError('Official component revision or checksum differs')
    if (reference['tokens'] != template_manifest['tokens']
            or {k: v['shape'] for k, v in reference['inputs'].items()}
            != {k: v['shape'] for k, v in template_fixture['inputs'].items()}):
        raise ValueError('Reference input shapes differ from the static graph')
    if output.exists():
        raise ValueError('Use a new output directory')
    with weights.open('rb') as inp:
        header_size = struct.unpack('<Q', inp.read(8))[0]
        if header_size > 16 * 1024 * 1024:
            raise ValueError('Unexpected component header size')
        header = json.loads(inp.read(header_size))
    prefix = f'layers.{block}.'
    expected_names = {prefix + name for name, shape in BINDINGS.values()}
    if set(header) - {'__metadata__'} != expected_names:
        raise ValueError('Component does not contain exactly the 11 expected block tensors')
    output.mkdir(parents=True)
    sections = []
    fp32_stream = hashlib.sha256()
    with weights.open('rb') as inp, (output / 'block.ncnn.bin.partial').open('xb') as out:
        for line in graph.splitlines()[2:]:
            fields = line.split()
            if fields[0] not in ['RMSNorm', 'Gemm']:
                continue
            name, shape = BINDINGS[fields[1]]
            item = header[prefix + name]
            begin, end = item['data_offsets']
            if item['dtype'] != 'BF16' or item['shape'] != shape or end - begin != math.prod(shape) * 2:
                raise ValueError(f'Unexpected tensor shape, dtype or byte count: {name}')
            norm = fields[0] == 'RMSNorm'
            offset = out.tell()
            if not norm:
                out.write(struct.pack('<I', 0x01348B83))
                fp32_stream.update(b'\0' * 4)
            inp.seek(8 + header_size + begin)
            remaining = end - begin
            tensor_hash = hashlib.sha256()
            while remaining:
                raw = inp.read(min(remaining, 1024 * 1024))
                if not raw or len(raw) % 2:
                    raise ValueError(f'Truncated BF16 tensor: {name}')
                restored = (np.frombuffer(raw, '<u2').astype('<u4') << 16).tobytes()
                if not np.isfinite(np.frombuffer(restored, '<f4')).all():
                    raise ValueError(f'Non-finite weight: {name}')
                tensor_hash.update(raw)
                fp32_stream.update(restored)
                out.write(restored if norm else raw)
                remaining -= len(raw)
            if not norm:
                out.write(b'\0' * ((-(end - begin)) % 4))
            sections.append({'layer': fields[1], 'tensor': prefix + name, 'shape': shape,
                             'source_sha256': tensor_hash.hexdigest(), 'output_offsets': [offset, out.tell()],
                             'storage': 'fp32_affine' if norm else 'bf16_modelbin'})
    if len(sections) != 11:
        raise ValueError('Incomplete weight stream')
    (output / 'block.ncnn.bin.partial').rename(output / 'block.ncnn.bin')
    (output / 'block.ncnn.param').write_text(graph)
    # Retain an independently generated official-reference fixture. No weights
    # or reference outputs are inherited from the template's different block.
    reference_names = ['fixture.json']
    for item in [*reference['inputs'].values(), reference['expected']]:
        name = item['file']
        if Path(name).name != name or sha256(fixture / name) != item['sha256']:
            raise ValueError('Reference tensor checksum mismatch')
        if (fixture / name).stat().st_size != math.prod(item['shape']) * 4:
            raise ValueError('Reference tensor byte count mismatch')
        reference_names.append(name)
    for name in reference_names:
        shutil.copyfile(fixture / name, output / name)
    info = {'method': 'direct_official_bf16', 'schema_version': 1, 'block': block,
            'component_sha256': source['sha256'], 'builder_sha256': sha256(__file__),
            'template_manifest_sha256': sha256(template / 'model.json'), 'graph_contract_sha256': GRAPH_SHA256,
            'reconstructed_fp32_sha256': fp32_stream.hexdigest(), 'sections': sections,
            'stored_bytes': (output / 'block.ncnn.bin').stat().st_size,
            'scope': 'Direct lossless file conversion; ModelBin expands weights during loading',
            'elapsed_seconds': time.perf_counter() - started}
    (output / 'direct-conversion.json').write_text(json.dumps(info, indent=2) + '\n')
    names = ['block.ncnn.param', 'block.ncnn.bin', 'direct-conversion.json', *reference_names]
    manifest = {'schema_version': 1, 'tokens': reference['tokens'], 'block': block,
                'official_model_revision': source['revision'], 'weights_sha256': source['sha256'],
                'ncnn_revision': lock['ncnn']['revision'], 'weight_storage': info,
                'files': {name: sha256(output / name) for name in names}}
    (output / 'model.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.template, args.weights, args.fixture, args.output)))
