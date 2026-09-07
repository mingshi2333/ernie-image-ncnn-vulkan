#!/usr/bin/env python3
"""Compare a real PE block's incremental native cache against full official attention."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from prepare_block import ROOT, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--runner', type=Path, default=ROOT/'build/ernie-pe-block-runner')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--chunk-plan', help='Optional real-token chunks summing to17, e.g.8,8,1')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if not 1 <= args.threads <= 256: p.error('Threads must be in [1,256]')
    if args.chunk_plan is None and args.threads != 4: p.error('--threads requires --chunk-plan')
    if args.output.exists(): p.error('Use new output directory')
    manifest = json.loads((args.model/'model.json').read_text())
    for name, digest in manifest['files'].items():
        if Path(name).name != name or sha256(args.model/name) != digest:
            raise ValueError('PE model checksum differs')
    fixture = json.loads((args.model/'fixture.json').read_text())
    for entry in [*fixture['inputs'].values(), fixture['expected']]:
        if sha256(args.model/entry['file']) != entry['sha256']:
            raise ValueError('PE reference tensor differs')
    args.output.mkdir(parents=True)
    runner = args.output/'runner.snapshot'; shutil.copy2(args.runner, runner)
    command = [str(runner.resolve()), str(args.model.resolve()), str(args.model.resolve()),
               str((args.output/'actual.f32').resolve())]
    if args.chunk_plan is not None:
        command.extend([args.chunk_plan, str(args.threads)])
    with (args.output/'native.log').open('w') as log:
        rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180).returncode
    result = {'passed': False, 'return_code': rc, 'scope': fixture['scope'], 'command': command,
              'model_manifest_sha256': sha256(args.model/'model.json'),
              'fixture_sha256': sha256(args.model/'fixture.json'), 'runner_sha256': sha256(runner),
              'validator_sha256': sha256(__file__)}
    if rc == 0:
        expected = np.fromfile(args.model/fixture['expected']['file'], '<f4').astype(np.float64)
        actual = np.fromfile(args.output/'actual.f32', '<f4').astype(np.float64)
        if expected.shape != actual.shape or not np.isfinite(actual).all():
            raise ValueError('Invalid PE output')
        error = actual-expected; gate = fixture['gates']
        nrmse = float(np.linalg.norm(error)/max(np.linalg.norm(expected), 1e-30))
        maximum = float(np.abs(error).max()); limit = gate['atol']+gate['rtol']*float(np.abs(expected).max())
        result.update(nrmse=nrmse, max_abs_error=maximum, max_abs_limit=limit, nrmse_limit=gate['nrmse'],
                      passed=nrmse <= gate['nrmse'] and maximum <= limit,
                      actual_sha256=sha256(args.output/'actual.f32'))
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
