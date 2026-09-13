#!/usr/bin/env python3
"""Derive row-broadcast SDPA from authenticated ncnn sources, outside the checkout.

The native shader registry and pipeline cache are preserved. All four attention
shaders receive the actual mask height; dense/causal masks keep their indexing.
The preceding BF16 compatibility derivation remains part of the source identity.
"""
import argparse
import hashlib
import json
from pathlib import Path

SHADERS = {
    'sdpa_cross': 'a6e6d5caaa411e4253a01a57b5fef461eeb1d9e634e3043920b9c6ad7dca4b14',
    'sdpa_cross_cm': '7c995bf4f9a17ac7ecc7a20499a23b6edfdf092707938b4e3a3f04840110f35a',
    'sdpa_fa': '89d31199eb9d35a28c86067105d7cace2a2fda6bb8d24702a48411c19fe22fa0',
    'sdpa_fa_cm': '956b83ef564b683f53ecb6f41d51416ef08bc306b7abb86a74a0188b82387743',
}
GPU_SHA = '12e5066a878a1d0d75f771de22100c8a54659c3c8a5de80d885da5c9190bfe7c'
BF16_SDPA_SHA = 'd7900bf93011ccbb8ebb3ef843b312922d169f5c64770de8a05fecf84f8d7e0a'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def authenticated(path, expected):
    text = Path(path).read_text()  # canonical LF, including Windows checkouts
    if digest(text) != expected:
        raise ValueError(f'Review changed ncnn source before deriving broadcast masks: {path}')
    return text


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError(f'Expected {count} occurrences of {before!r}')
    return text.replace(before, after)


def shader(name, text):
    text = replace(text, '    int mask_cstep;\n', '    int mask_cstep;\n    int mask_h;\n')
    if name == 'sdpa_cross':
        text = replace(text, 'uvec4 mi4 = gy4 * psc(N) + gx * 4;',
                       'uvec4 mi4 = min(gy4, uvec4(p.mask_h - 1)) * psc(N);')
        # The final vector tile can straddle N or M. Its unused lanes must not
        # read beyond even a one-row descriptor. Valid output lanes are unchanged.
        for row in 'rgba':
            for col in range(4):
                text = replace(text, f'mi4.{row} + {col}',
                               f'mi4.{row} + min(gx * 4 + {col}, uint(psc(N) - 1))')
    elif name == 'sdpa_cross_cm':
        text = replace(text, 'uvec4 ci4 = gm * psc(GN) + gn * 8 + uvec4(0, 1, 2, 3);',
                       'uvec4 ci4 = min(gm, uint(p.mask_h - 1)) * psc(GN) + gn * 8 + uvec4(0, 1, 2, 3);')
    else:
        text = replace(text, 'mask_head * p.mask_cstep + gm * p.dst_seqlen + gn;',
                       'mask_head * p.mask_cstep + min(gm, uint(p.mask_h - 1)) * p.dst_seqlen + gn;',
                       2 if name == 'sdpa_fa_cm' else 1)
    return text


def sdpa(text):
    text = replace(text, 'std::vector<vk_constant_type> constants(11);',
                   'std::vector<vk_constant_type> constants(12);', 2)
    text = replace(text, 'std::vector<vk_constant_type> constants(13);',
                   'std::vector<vk_constant_type> constants(14);', 2)
    text = replace(text, 'constants[12].i = attn_mask_blob.cstep;',
                   'constants[12].i = attn_mask_blob.cstep;\n            constants[13].i = attn_mask_blob.h;', 2)
    text = replace(text, 'constants[10].i = attn_mask_blob.cstep;',
                   'constants[10].i = attn_mask_blob.cstep;\n        constants[11].i = attn_mask_blob.h;')
    return replace(text, 'constants[10].i = 0; // mask_cstep',
                   'constants[10].i = 0; // mask_cstep\n        constants[11].i = 0; // no mask')


def write_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def derive(source, bf16_source, registry, output):
    originals = {name: authenticated(source / 'src/layer/vulkan/shader' / (name + '.comp'), sha)
                 for name, sha in SHADERS.items()}
    gpu = authenticated(source / 'src/gpu.cpp', GPU_SHA)
    attention = authenticated(bf16_source, BF16_SDPA_SHA)
    registry_text = registry.read_text()
    outputs = {}
    for name, original in originals.items():
        modified = shader(name, original)
        outputs[name + '.comp'] = modified
        # Same byte-array interface as ncnn's generated shader registry, so
        # compilation, precision variants and pipeline caching stay native.
        data = ','.join(f'0x{b:02x}' for b in modified.encode())
        outputs[name + '.comp.hex.h'] = f'static const char {name}_comp_data[] = {{{data}}};\n'
        registry_text = replace(registry_text, f'"layer/vulkan/shader/{name}.comp.hex.h"',
                                f'"{name}.comp.hex.h"')
    outputs['layer_shader_spv_data.h'] = registry_text
    # The adjacent derived registry wins the quoted include search. All other
    # includes still resolve through the original ncnn source/build directories.
    outputs['gpu.cpp'] = gpu
    outputs['sdpa_vulkan.cpp'] = sdpa(attention)
    output.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        write_changed(output / name, text)
    write_changed(output / 'provenance.json', json.dumps({
        'source_sha256': {**SHADERS, 'gpu': GPU_SHA, 'bf16_sdpa': BF16_SDPA_SHA},
        'derived_sha256': {name: digest(text) for name, text in outputs.items()},
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'bf16-source', 'registry', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    derive(args.source, args.bf16_source, args.registry, args.output)
