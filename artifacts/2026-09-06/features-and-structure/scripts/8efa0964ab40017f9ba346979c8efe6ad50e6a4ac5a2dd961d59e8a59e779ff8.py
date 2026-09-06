#!/usr/bin/env python3
"""Prepare and seal a checked pnnx block for the ERNIE runtime, without re-exporting weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for data in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    fixture = json.loads((source / 'fixture.json').read_text())
    conversion = json.loads((source / 'conversion.json').read_text())
    if (conversion['return_code'] or conversion.get('unsupported_diagnostics', [])
            or conversion.get('unconverted_operators', []) or conversion.get('unsafe_rotary_fusion', True)
            or conversion.get('sdpa_count') != 1 or conversion.get('rmsnorm_count') != 4):
        raise ValueError('Source conversion has not passed the required structural checks')
    if output.exists():
        raise ValueError('Use a new output directory to preserve previous attempts')
    tokens = fixture['tokens']
    lines = (source / 'block.ncnn.param').read_text().splitlines()
    modified, gelu_count, reshape_count = [], 0, 0
    for line in lines:
        fields = line.split()
        if fields and fields[0] == 'GELU':
            if fields[2:4] != ['1', '1'] or (len(fields) > 6 and fields[6:] != ['0=0']):
                raise ValueError('Unexpected GELU signature')
            fields[0] = 'ErnieGELU'
            modified.append(' '.join(fields[:6]))
            gelu_count += 1
        elif fields and fields[0] == 'ExpandDims':
            if fields[2:4] != ['1', '1'] or fields[4] not in ['in7', 'in8'] or fields[6:] != ['-23303=1,0']:
                raise ValueError('Unexpected ExpandDims signature')
            # [S,128] -> [1,S,128] after batch removal. Reshape stays on Vulkan.
            fields[0] = 'Reshape'
            modified.append(' '.join(fields[:6] + ['0=128', f'1={tokens}', '2=1']))
            reshape_count += 1
        else:
            modified.append(line)
    if gelu_count != 1 or reshape_count != 2:
        raise ValueError('Export graph differs from the reviewed block structure')
    output.mkdir(parents=True)
    filenames = ['fixture.json', 'conversion.json', 'conversion.log', 'block.ncnn.bin',
                 fixture['expected']['file'], *(item['file'] for item in fixture['inputs'].values())]
    for name in filenames:
        try:
            os.link(source / name, output / name)
        except OSError:
            shutil.copyfile(source / name, output / name)
    shutil.copyfile(source / 'block.ncnn.param', output / 'block.pnnx-original.param')
    (output / 'block.ncnn.param').write_text('\n'.join(modified) + '\n')
    manifest = {
        'schema_version': 1, 'tokens': tokens, 'block': fixture['block'],
        'official_model_revision': fixture['official_model_revision'],
        'weights_sha256': fixture['weights_sha256'],
        'ncnn_revision': json.loads((ROOT / 'sources.lock.json').read_text())['ncnn']['revision'],
        'changes': ['GELU -> ErnieGELU: erf-form GPU implementation',
                    'Two constant-table ExpandDims -> equivalent Vulkan Reshape'],
        'prepare_source_sha256': sha256(__file__),
        'gelu_source_sha256': sha256(ROOT / 'src/ernie_gelu.cpp'),
        'files': {name: sha256(output / name) for name in [*filenames, 'block.ncnn.param', 'block.pnnx-original.param']},
    }
    (output / 'model.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output)))
