#!/usr/bin/env python3
"""Compare shared pipeline cache with a retained identical-input baseline."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_block import ROOT, sha256
from validate_block_sequence import run
from validate_dit_block import verify as verify_block
from validate_dit_heads import verify as verify_head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runner', type=Path, default=ROOT / 'build/ernie-block-sequence-runner')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new comparison directory')
    baseline = json.loads((args.baseline / 'matrix.json').read_text())
    fixture_dir = args.baseline / 'reference'
    fixture = json.loads((fixture_dir / 'fixture.json').read_text())
    if len(baseline) != 4 or any(x.get('return_code') != 0 for x in baseline):
        parser.error('Require a completed four-configuration baseline')
    args.output.mkdir(parents=True)
    runner = args.output / 'ernie-dit-runner.snapshot'
    shutil.copy2(args.runner, runner)
    results = []
    for original in baseline:
        if original['backend'] != 'vulkan':
            continue
        if original['runtime'].get('shared_pipeline_cache', False):
            raise ValueError('Baseline already uses shared pipeline cache')
        if original['fixture_sha256'] != sha256(fixture_dir / 'fixture.json'):
            raise ValueError('Baseline fixture has changed')
        command = original['command']
        models = [Path(command[i + 1]) for i, value in enumerate(command) if value == '--model']
        for model in models:
            verify_block(model, model)
        extra = []
        heads = []
        for flag in ('--input-head', '--output-head', '--width', '--height', '--text-tokens'):
            if flag in command:
                value = command[command.index(flag) + 1]
                extra += [flag, value]
                if flag.endswith('-head'):
                    verify_head(Path(value))
                    heads.append(Path(value))
        ordered = [heads[0], *models, heads[1]] if heads else models
        if [sha256(path / 'model.json') for path in ordered] != fixture['model_manifests']:
            raise ValueError('Models differ from the original reference')
        precision = original['precision']
        directory = args.output / f'vulkan-{precision}'
        result = run(models, fixture_dir, fixture, directory, runner, 'vulkan', precision, 'stream', extra)
        exact = result.get('actual_sha256') == original.get('actual_sha256')
        result['baseline_result_sha256'] = sha256(args.baseline / 'matrix.json')
        result['matches_baseline_exactly'] = exact
        result['passed'] &= exact and result.get('runtime', {}).get('shared_pipeline_cache', False)
        old_runtime, new_runtime = original['runtime'], result.get('runtime', {})
        result['timing_comparison'] = {
            'baseline_load_seconds': sum(old_runtime['load_seconds']),
            'shared_load_seconds': sum(new_runtime.get('load_seconds', [])),
            'baseline_compute_seconds': sum(old_runtime['compute_seconds']),
            'shared_compute_seconds': sum(new_runtime.get('compute_seconds', [])),
            'scope': 'One before/after observation, same weights/input/precision; load includes pipeline creation and weight preparation; not end-to-end image speedup',
        }
        (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        results.append(result)
        (args.output / 'matrix.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({'precision': precision, 'passed': result['passed'], 'exact': exact,
                          **result['timing_comparison']}), flush=True)
    return 0 if len(results) == 3 and all(item['passed'] for item in results) else 1


if __name__ == '__main__':
    main()
