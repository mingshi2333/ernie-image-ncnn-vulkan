#!/usr/bin/env python3
"""Losslessly store official BF16-derived Gemm weights in ncnn's native BF16 ModelBin format."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import numpy as np
from prepare_block import sha256

NO_WEIGHTS = {'Input', 'Split', 'BinaryOp', 'Reshape', 'Permute', 'Slice', 'Concat', 'SDPA', 'ErnieGELU'}
BF16_TAG = 0x01348B83


def pack(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = json.loads((source / 'model.json').read_text())
    if output.exists():
        raise ValueError('Use a new destination')
    if sha256(source / 'block.ncnn.bin') != manifest['files']['block.ncnn.bin']:
        raise ValueError('Source weight checksum mismatch')
    output.mkdir(parents=True)
    reconstructed = hashlib.sha256()
    gemm_count = norm_count = 0
    with (source / 'block.ncnn.bin').open('rb') as inp, (output / 'block.ncnn.bin.partial').open('xb') as out:
        for line in (source / 'block.ncnn.param').read_text().splitlines()[2:]:
            fields = line.split()
            if not fields:
                continue
            kind = fields[0]
            params = dict(item.split('=', 1) for item in fields[4 + int(fields[2]) + int(fields[3]):])
            if kind == 'RMSNorm':
                if params.get('2', '1') != '1':
                    raise ValueError('Only affine RMSNorm is supported')
                count = int(params['0'])
                raw = inp.read(count * 4)
                if len(raw) != count * 4:
                    raise ValueError('Truncated normalization weights')
                out.write(raw)
                reconstructed.update(raw)
                norm_count += 1
            elif kind == 'Gemm':
                if (params.get('4', '0') != '0' or params.get('5', '0') != '1'
                        or params.get('10') != '-1' or params.get('18', '0') != '0'):
                    raise ValueError('Only the reviewed constant-B, bias-free Gemm is supported')
                count = int(params['8']) * int(params['9'])
                original_tag = inp.read(4)
                if original_tag != b'\0\0\0\0':
                    raise ValueError('Expected an unquantized FP32 weight section')
                out.write(struct.pack('<I', BF16_TAG))
                reconstructed.update(original_tag)
                remaining = count
                while remaining:
                    take = min(remaining, 262144)
                    raw = inp.read(take * 4)
                    if len(raw) != take * 4:
                        raise ValueError('Truncated Gemm weights')
                    bits = np.frombuffer(raw, dtype='<u4')
                    if np.any(bits & 0xffff):
                        raise ValueError('Weights are not exactly representable as BF16; refusing lossy conversion')
                    packed = (bits >> 16).astype('<u2')
                    out.write(packed.tobytes())
                    restored = (packed.astype('<u4') << 16).tobytes()
                    reconstructed.update(restored)
                    remaining -= take
                out.write(b'\0' * ((-count * 2) % 4))
                gemm_count += 1
            elif kind not in NO_WEIGHTS:
                raise ValueError(f'Unsupported layer: {kind}')
        if inp.read(1) or gemm_count != 7 or norm_count != 4:
            raise ValueError('Weight stream did not match one complete reviewed block')
    if reconstructed.hexdigest() != manifest['files']['block.ncnn.bin']:
        raise ValueError('Lossless round-trip checksum failed')
    (output / 'block.ncnn.bin.partial').rename(output / 'block.ncnn.bin')
    for name, checksum in manifest['files'].items():
        if name == 'block.ncnn.bin':
            continue
        if Path(name).name != name or sha256(source / name) != checksum:
            raise ValueError(f'Source artifact checksum mismatch: {name}')
        try:
            os.link(source / name, output / name)
        except OSError:
            shutil.copyfile(source / name, output / name)
    manifest['weight_storage'] = {
        'format': 'BF16 ModelBin Gemm sections, FP32 affine RMSNorm sections',
        'lossless': True, 'original_bytes': (source / 'block.ncnn.bin').stat().st_size,
        'stored_bytes': (output / 'block.ncnn.bin').stat().st_size,
        'reconstructed_fp32_sha256': reconstructed.hexdigest(),
        'source_model_sha256': sha256(source / 'model.json'), 'packer_sha256': sha256(__file__),
        'scope': 'On-disk representation only. ModelBin expands to float32 during loading; this does not promise lower peak RAM/VRAM.'}
    manifest['files']['block.ncnn.bin'] = sha256(output / 'block.ncnn.bin')
    (output / 'model.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest['weight_storage']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pack(args.model, args.output)))
