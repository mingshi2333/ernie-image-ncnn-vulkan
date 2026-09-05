#!/usr/bin/env python3
"""Convert and validate an explicit range of official blocks, one block at a time."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time
from build_dit_weights import build
from fetch_component import fetch_component
from prepare_block import ROOT, sha256
from validate_dit_block import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=36, help='Exclusive upper bound')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--runner', type=Path, default=ROOT / 'build/ernie-block-runner')
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--keep-going-on-numerical-failure', action='store_true',
                        help='Retain failed gates and inspect later blocks; execution/conversion failures still stop')
    args = parser.parse_args()
    if not 0 <= args.start < args.end <= 36 or not 1 <= args.repeat <= 20:
        parser.error('Invalid block range or repeat count')
    if args.output.exists():
        parser.error('Use a new output directory')
    template, reference = verify(args.template, args.template)
    args.output.mkdir(parents=True)
    # Isolate the executable from concurrent rebuilds of the development tree.
    runner = args.output / 'ernie-block-runner.snapshot'
    shutil.copy2(args.runner, runner)
    state = {'schema_version': 1, 'start': args.start, 'end': args.end, 'tokens': template['tokens'],
             'template_sha256': sha256(args.template / 'model.json'), 'orchestrator_sha256': sha256(__file__),
             'scope': 'Independent blocks with synthetic activations; not a connected DiT or image generator',
             'source_lock_sha256': sha256(ROOT / 'sources.lock.json'), 'runner_sha256': sha256(runner),
             'keep_going_on_numerical_failure': args.keep_going_on_numerical_failure,
             'complete': False, 'passed': False, 'blocks': []}
    state_path = args.output / 'conversion-run.json'

    def persist():
        temporary = state_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)

    persist()
    for block in range(args.start, args.end):
        started = time.perf_counter()
        row = {'block': block, 'stage': 'fetch', 'passed': False}
        state['blocks'].append(row)
        directory = args.output / f'block-{block:02d}'
        directory.mkdir()
        persist()
        print(json.dumps({'block': block, 'stage': 'fetch'}), flush=True)
        try:
            weights = ROOT / f'models/official/dit-block-{block:02d}.safetensors'
            with (directory / 'fetch.log').open('w') as log, redirect_stdout(log):
                component = fetch_component(f'layers.{block}.', weights)
            row['component_sha256'] = component['sha256']
            row['stage'] = 'reference'
            persist()
            command = [sys.executable, str(ROOT / 'tools/export_dit_block.py'), '--block', str(block),
                       '--weights', str(weights), '--fixture-only', '--height', str(reference['grid'][0]),
                       '--width', str(reference['grid'][1]), '--text-tokens', str(reference['text_tokens']),
                       '--valid-text', str(reference['valid_text']), '--seed', str(reference['seed']),
                       '--output', str((directory / 'reference').resolve())]
            row['reference_command'] = command
            with (directory / 'reference.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            row['stage'] = 'convert'
            persist()
            info = build(args.template, weights, directory / 'reference', directory / 'runtime')
            row['conversion_seconds'] = info['elapsed_seconds']
            row['model_manifest_sha256'] = sha256(directory / 'runtime/model.json')
            row['stage'] = 'validate'
            persist()
            if args.cpu_only:
                command = [sys.executable, str(ROOT / 'tools/validate_dit_block.py'), '--model',
                           str((directory / 'runtime').resolve()), '--output', str((directory / 'validation').resolve()),
                           '--backend', 'cpu', '--repeat', str(args.repeat), '--runner', str(runner.resolve())]
            else:
                command = [sys.executable, str(ROOT / 'tools/run_block_matrix.py'), '--model',
                           str((directory / 'runtime').resolve()), '--output', str((directory / 'validation').resolve()),
                           '--repeat', str(args.repeat), '--trace-attention', '--runner', str(runner.resolve())]
            row['validation_command'] = command
            with (directory / 'validation.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2600)
            row.update(stage='complete', passed=True, elapsed_seconds=time.perf_counter() - started)
            persist()
            print(json.dumps(row), flush=True)
        except Exception as error:
            # Retain a truthful journal and all partials. Retry in a new run
            # directory after diagnosing the failed stage; never hide failures.
            row.update(failure=f'{type(error).__name__}: {error}', elapsed_seconds=time.perf_counter() - started)
            persist()
            print(json.dumps(row), flush=True)
            matrix_path = directory / 'validation/matrix.json'
            if (args.keep_going_on_numerical_failure and row['stage'] == 'validate'
                    and isinstance(error, subprocess.CalledProcessError) and matrix_path.is_file()):
                matrix = json.loads(matrix_path.read_text())
                if len(matrix) == 4 and all(item.get('return_code') == 0 for item in matrix):
                    # Complete executors with failed numerical gates are evidence,
                    # not a successful conversion certification. Never change gates.
                    continue
            return 1
    state.update(complete=True, passed=all(row['passed'] for row in state['blocks']))
    persist()
    return 0 if state['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
