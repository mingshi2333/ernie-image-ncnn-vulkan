#!/usr/bin/env python3
"""Measure the official PyTorch block's own backend/dtype spread on a saved fixture."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from export_dit_block import load_block, make_inputs, metrics, sha256, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    fixture = json.loads((args.fixture / 'fixture.json').read_text())
    if fixture['tokens'] > 512:
        parser.error('This calibration tool is limited to <=512 tokens to bound CUDA reference memory')
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    expected_entry = fixture['expected']
    if sha256(args.fixture / expected_entry['file']) != expected_entry['sha256']:
        raise ValueError('Reference checksum mismatch')
    expected = torch.from_numpy(np.fromfile(args.fixture / expected_entry['file'], dtype='<f4').reshape(expected_entry['shape']))
    inputs = []
    for key, entry in fixture['inputs'].items():
        if sha256(args.fixture / entry['file']) != entry['sha256']:
            raise ValueError('Fixture checksum mismatch')
        inputs.append(torch.from_numpy(np.fromfile(args.fixture / entry['file'], dtype='<f4').reshape(entry['shape'])))
    _, freqs = make_inputs(*fixture['grid'], fixture['text_tokens'], fixture['valid_text'], fixture['seed'])
    if not torch.equal(freqs[0, :, 0].cos(), inputs[7][0]) or not torch.equal(freqs[0, :, 0].sin(), inputs[8][0]):
        raise ValueError('Reconstructed reference positions changed')
    block, _ = load_block(ROOT / 'models/official' / f'dit-block-{fixture["block"]:02d}.safetensors', fixture['block'])
    results = {'fixture_sha256': sha256(args.fixture / 'fixture.json'), 'torch_build': torch.__version__,
               'source_sha256': sha256(__file__), 'tf32': False, 'reference': 'Saved official CPU FP32 output',
               'scope': 'Post-candidate calibration only; does not change predeclared ncnn gates', 'cases': [], 'complete': False}
    for name, dtype in [('fp32', torch.float32), ('fp16', torch.float16), ('bf16', torch.bfloat16)]:
        if not torch.cuda.is_available():
            results['cases'].append({'precision': name, 'status': 'unavailable'})
            continue
        # Reconstruct each dtype from original FP32 weights, avoiding serial rounding.
        model = block.to('cuda', dtype=dtype)
        x = inputs[0].to('cuda', dtype=dtype).transpose(0, 1)
        temb = [item.to('cuda', dtype=dtype) for item in inputs[1:7]]
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = model(x, freqs.to('cuda'), temb, attention_mask=inputs[9].to('cuda', dtype=dtype)[None, None]).transpose(0, 1)
        torch.cuda.synchronize()
        results['cases'].append({'precision': name, 'status': 'measured', 'seconds': time.perf_counter() - start,
                                 **metrics(expected, out.cpu().float())})
        (args.output / 'result.json').write_text(json.dumps(results, indent=2) + '\n')
        del out, x, temb, model, block
        torch.cuda.empty_cache()
        block, _ = load_block(ROOT / 'models/official' / f'dit-block-{fixture["block"]:02d}.safetensors', fixture['block'])
    results['complete'] = True
    (args.output / 'result.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
