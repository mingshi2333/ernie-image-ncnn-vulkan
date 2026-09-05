#!/usr/bin/env python3
"""Run a sealed block through CPU FP32 and device-resident Vulkan precision checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--runner', type=Path, default=ROOT / 'build/ernie-block-runner')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--host-weights', action='store_true')
    parser.add_argument('--trace-attention', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    results = []
    for backend, precision in [('cpu', 'fp32'), ('vulkan', 'fp32'), ('vulkan', 'fp16'), ('vulkan', 'bf16')]:
        output = args.output / f'{backend}-{precision}'
        command = [sys.executable, str(ROOT / 'tools/validate_dit_block.py'), '--model', str(args.model.resolve()),
                   '--output', str(output.resolve()), '--backend', backend, '--precision', precision,
                   '--repeat', str(args.repeat), '--runner', str(args.runner.resolve())]
        if backend == 'vulkan':
            command.append('--device-io')
            if args.trace_attention:
                command.append('--trace-attention')
            if args.host_weights:
                command.append('--host-weights')
        process = subprocess.run(command, capture_output=True, text=True)
        (args.output / f'{backend}-{precision}.log').write_text(process.stdout + process.stderr)
        result = json.loads((output / 'result.json').read_text()) if (output / 'result.json').exists() else {'passed': False}
        result['validator_exit'] = process.returncode
        results.append(result)
        print(json.dumps({key: result.get(key) for key in ['backend', 'precision', 'passed', 'failure', 'nrmse', 'max_rss_kib']}), flush=True)
    (args.output / 'matrix.json').write_text(json.dumps(results, indent=2) + '\n')
    return 0 if all(item['passed'] and item['validator_exit'] == 0 for item in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
