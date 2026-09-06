#!/usr/bin/env python3
"""Compare all native greedy PE logits and IDs with a sealed official oracle."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from prepare_block import ROOT, sha256
from validate_pe_tokenizer import validate_development_batch
from transformers import AutoTokenizer

# Declared before real full-model runs; no sampling-seed equivalence is assumed.
GATES = dict(nrmse=.0002, atol=.0002, rtol=.0002)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runner', type=Path, default=ROOT/'build/ernie-pe-runner')
    p.add_argument('--batch-contract', type=Path, help='Require reference membership in this frozen batch')
    p.add_argument('--tokenizer', type=Path, default=ROOT/'models/pe-tokenizer')
    args = p.parse_args()
    if args.output.exists(): p.error('Use a new output directory')
    reference = args.reference.resolve()
    metadata = json.loads((reference/'reference.json').read_text())
    batch_case = None
    if args.batch_contract:
        batch = json.loads(args.batch_contract.read_text())
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        validate_development_batch(batch, tokenizer)
        if metadata.get('batch_suite') != batch['suite']:
            raise ValueError('PE reference batch suite differs')
        matches = [case for case in batch['cases'] if case['id'] == metadata.get('batch_case_id')]
        if len(matches) != 1:
            raise ValueError('PE reference case is absent from the frozen batch')
        batch_case = matches[0]
        expected = {key: metadata.get(key) for key in ('input_prompt','width','height','max_tokens')}
        actual = dict(input_prompt=batch_case['prompt'], width=batch_case['width'],
                      height=batch_case['height'], max_tokens=batch_case['max_tokens'])
        if expected != actual or metadata.get('batch_input_ids_sha256') != batch_case['input_ids_sha256']:
            raise ValueError('PE reference input differs from the frozen batch case')
    for name, digest in metadata['files'].items():
        if Path(name).name != name or sha256(reference/name) != digest:
            raise ValueError('PE reference file differs')
    out = args.output.resolve(); out.mkdir(parents=True)
    runner = out/'ernie-pe-runner'; shutil.copy2(args.runner, runner)
    prompt = out/'prompt.txt'; prompt.write_bytes(metadata['input_prompt'].encode())
    command = [str(runner), str(args.model.resolve()), str(prompt), str(metadata['width']),
               str(metadata['height']), str(metadata['max_tokens']), str(out/'native'), 'greedy']
    started = time.monotonic()
    with (out/'native.log').open('w') as log:
        rc = subprocess.run(['/usr/bin/time', '-v', *command], stdout=log, stderr=subprocess.STDOUT).returncode
    rows = []
    native = out/'native'
    for index in range(metadata['generated_tokens']):
        path = native/f'logits-{index}.f32'
        if not path.is_file(): break
        expected = np.fromfile(reference/path.name, '<f4').astype(np.float64)
        actual = np.fromfile(path, '<f4').astype(np.float64)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            rows.append(dict(token=index, passed=False, reason='Invalid logits')); continue
        error = actual-expected
        nrmse = float(np.sqrt(np.mean(error**2))/max(np.sqrt(np.mean(expected**2)), 1e-12))
        maximum = float(np.max(np.abs(error)))
        limit = GATES['atol']+GATES['rtol']*float(np.max(np.abs(expected)))
        rows.append(dict(token=index, nrmse=nrmse, max_abs_error=maximum, max_abs_limit=limit,
                         passed=nrmse <= GATES['nrmse'] and maximum <= limit))
    same = {name: (native/name).is_file() and (native/name).read_bytes() == (reference/name).read_bytes()
            for name in ('input-ids.txt', 'generated-ids.txt', 'enhanced.txt')}
    passed = rc == 0 and all(same.values()) and len(rows) == metadata['generated_tokens'] and all(x['passed'] for x in rows)
    result = dict(passed=passed, return_code=rc, elapsed_seconds=time.monotonic()-started,
                  gates=GATES, exact=same, logits=rows, runner_sha256=sha256(runner), validator_sha256=sha256(__file__),
                  model_manifest_sha256=sha256(args.model/'manifest.json'),
                  reference_manifest_sha256=sha256(reference/'reference.json'),
                  native_files={p.name: sha256(p) for p in sorted(native.iterdir()) if p.is_file()} if native.is_dir() else {})
    if batch_case:
        result.update(batch_suite=batch['suite'], batch_case_id=batch_case['id'],
                      batch_acceptance_status=('official_and_native_complete' if passed else 'incomplete'))
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': passed, 'exact': same, 'logits_passed': sum(x['passed'] for x in rows),
                      'logits_total': metadata['generated_tokens']}), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__': main()
