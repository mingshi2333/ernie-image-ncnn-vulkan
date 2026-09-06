#!/usr/bin/env python3
"""Reuse lossless text weights with a separately exported, verified static graph."""
import argparse
import hashlib
import json
from pathlib import Path
from package_model import sha256

GRAPH_SHA256 = '67881d48031e178fcb5653f749d17895a9470d03adf5bf8098dd5bbff952ab36'
SHAPES = {**{f'gemm_{i}': '7' for i in range(7)},
          **{f'reshape_{i}': '2' for i in (10, 11, 12)},
          'reshape_13': '1', 'unsqueeze_18': '1', 'unsqueeze_19': '1'}


def graph_hash(graph, tokens):
    if not 1 <= tokens <= 2048:
        raise ValueError('Unsupported text bucket')
    lines, seen = [], set()
    for line in graph.splitlines():
        fields = line.split()
        if len(fields) > 1 and fields[1] in SHAPES:
            layer = fields[1]
            key = SHAPES[layer]
            if layer in seen or fields.count(f'{key}={tokens}') != 1:
                raise ValueError('Unexpected static text dimension')
            fields[fields.index(f'{key}={tokens}')] = f'{key}=TOKENS'
            seen.add(layer)
        lines.append(' '.join(fields))
    if seen != set(SHAPES):
        raise ValueError('Incomplete text graph')
    return hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()


def verified(path):
    manifest = json.loads((path/'model.json').read_text())
    if any(Path(name).name != name or sha256(path/name) != digest for name, digest in manifest['files'].items()):
        raise ValueError('Text source checksum differs')
    if graph_hash((path/'text.ncnn.param').read_text(), manifest['tokens']) != GRAPH_SHA256:
        raise ValueError('Text source graph differs from reviewed topology')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='All 25 lossless block directories')
    parser.add_argument('--template', type=Path, required=True, help='Independent export_text_block.py export of block zero')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    template = verified(args.template)
    if template['block'] != 0:
        raise ValueError('Require first-block reference template')
    base0 = verified(args.source/'block-00')
    if sha256(args.template/'text.ncnn.bin') != base0['reconstructed_fp32_sha256']:
        raise ValueError('Target export does not reconstruct the same FP32 weights')
    args.output.mkdir(parents=True)
    graph = (args.template/'text.ncnn.param').read_bytes()
    for index in range(25):
        source = args.source/f'block-{index:02d}'
        base = verified(source)
        if base['block'] != index or not base.get('reconstructed_fp32_sha256'):
            raise ValueError('Expected ordered lossless BF16 blocks')
        out = args.output/f'block-{index:02d}'
        out.mkdir()
        (out/'text.ncnn.param').write_bytes(graph)
        (out/'text.ncnn.bin').symlink_to((source/'text.ncnn.bin').resolve())
        item = {**base, 'tokens': template['tokens'], 'graph_sha256': sha256(out/'text.ncnn.param'),
                'graph_contract_sha256': GRAPH_SHA256, 'builder_sha256': sha256(__file__),
                'template_manifest_sha256': sha256(args.template/'model.json'),
                'base_manifest_sha256': sha256(source/'model.json'),
                'files': {name: sha256(out/name) for name in ('text.ncnn.param', 'text.ncnn.bin')}}
        (out/'model.json').write_text(json.dumps(item, indent=2)+'\n')
        print(json.dumps({'block': index, 'tokens': template['tokens']}), flush=True)


if __name__ == '__main__':
    main()
