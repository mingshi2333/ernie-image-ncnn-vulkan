#!/usr/bin/env python3
"""Recheck saved trajectories and report the first failure in execution order.

This is a diagnostic over development/historical runs. It never promotes a
reference-embedding or teacher-forced result to native generation acceptance.
"""
import argparse
import hashlib
import json
import math
import numpy as np
from pathlib import Path

from collect_parity_evidence import audit


def canonical_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def first_failed_boundary(rows):
    for row in rows:
        if not isinstance(row.get('boundary'), str) or not row['boundary']:
            raise ValueError('Every boundary needs a nonempty name')
        if type(row.get('passed')) is not bool:
            raise ValueError('Missing or ambiguous boundary verdict')
    return next((row['boundary'] for row in rows if not row['passed']), None)


def validate_rows(rows):
    names = set()
    for row in rows:
        if row['boundary'] in names:
            raise ValueError('Duplicate execution boundary')
        names.add(row['boundary'])
        if row.get('step') is not None and (type(row['step']) is not int or row['step'] < 0):
            raise ValueError('Invalid denoising step')
        if row.get('stage') not in ('text', 'conditioning', 'initial', 'dit', 'scheduler', 'unpack', 'vae'):
            raise ValueError('Unknown execution stage')
        if row.get('precision') not in ('fp32', 'fp16', 'bf16'):
            raise ValueError('Missing execution precision')
        for key in ('input_sha256', 'reference_sha256', 'candidate_sha256'):
            digest = row.get(key)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid tensor identity: ' + key)
        metrics = row.get('metrics', {})
        for key in ('max_abs_error', 'nrmse', 'reference_max_abs'):
            value = metrics.get(key)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid boundary metric: ' + key)
    first_failed_boundary(rows)


def ordered_boundaries(steps):
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError('Invalid trajectory length')
    result = [('text', None, 'text'), ('padded-text', None, 'conditioning')]
    result += [(f'constant-{i}', None, 'conditioning') for i in range(3)]
    result += [('initial', None, 'initial')]
    for step in range(steps):
        result += [(f'prediction-{step}', step, 'dit'), (f'step-{step}', step, 'scheduler')]
    return result + [('final', steps-1, 'scheduler'), ('unpacked', None, 'unpack'), ('decoded', None, 'vae')]


def conditioning_identity(result, fixture):
    source=result.get('conditioning_source','native_text_encoder')
    if source not in ('native_text_encoder','saved_reference_diagnostic','saved_candidate_diagnostic'):
        raise ValueError('Unknown conditioning source')
    if source!='saved_candidate_diagnostic':
        if result.get('diagnostic_embeddings') is not None:raise ValueError('Candidate metadata has wrong conditioning scope')
        return source,None
    meta=result.get('diagnostic_embeddings',{})
    shape=[1,len(fixture['ids']),3072]
    if (result.get('native_acceptance_eligible') is not False or meta.get('dtype')!='<f4'
        or meta.get('shape')!=shape or fixture['inputs']['text']['shape']!=shape
        or meta.get('snapshot_file')!='diagnostic-embeddings.f32'
        or meta.get('token_ids_sha256')!=canonical_sha256(fixture['ids'])
        or meta.get('source_sha256')!=meta.get('sha256')):
        raise ValueError('Invalid saved candidate conditioning contract')
    digest=meta.get('sha256')
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid saved candidate SHA256')
    text=[x for x in result['comparisons'] if x['tensor']=='text']
    if len(text)!=1 or text[0]['sha256']!=digest:
        raise ValueError('Measured text is not the saved candidate')
    return source,meta


def verify_candidate_snapshot(directory,result,fixture):
    source,meta=conditioning_identity(result,fixture)
    if meta is None:return
    directory=Path(directory).resolve();path=directory/meta['snapshot_file'];command=result['command']
    if (command.count('--embeddings')!=1 or '--pe-model' in command or command.index('--embeddings')+1>=len(command)
        or Path(command[command.index('--embeddings')+1]).resolve()!=path.resolve() or path.is_symlink()):
        raise ValueError('Runner did not read the local candidate snapshot')
    if path.stat().st_size!=int(np.prod(meta['shape']))*4:
        raise ValueError('Candidate snapshot shape differs')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=meta['sha256'] or not np.isfinite(np.frombuffer(raw,dtype='<f4')).all():
        raise ValueError('Candidate snapshot bytes or finiteness differ')
    if (directory/'trace/text.f32').read_bytes()!=raw:
        raise ValueError('Actual trace text differs from candidate snapshot')

def build_rows(result, fixture):
    """Require every saved tensor, with explicit data-dependency identities."""
    conditioning_source,diagnostic_input=conditioning_identity(result,fixture)
    comparisons = result['comparisons']
    measured = {row['tensor']: row for row in comparisons}
    order = ordered_boundaries(fixture['steps'])
    if len(measured) != len(comparisons) or set(measured) != {name for name, _, _ in order}:
        raise ValueError('Missing, duplicate or unexpected trajectory tensor')
    expected = dict(fixture['inputs'])
    for step, entries in enumerate(fixture['outputs']):
        expected.update({f'{name}-{step}': entry for name, entry in entries.items()})
    expected.update(fixture['final'])
    if set(expected) != set(measured):
        raise ValueError('Reference trajectory boundaries differ')
    rows = []
    for name, step, stage in order:
        m = measured[name]
        if name == 'text':
            inputs = {'token_ids': canonical_sha256(fixture['ids']),
                      'package': result['package_manifest_sha256'],
                      'conditioning_source': conditioning_source}
            if diagnostic_input:inputs['saved_candidate_embeddings']=diagnostic_input['sha256']
        elif name == 'padded-text':
            inputs = {'text': measured['text']['sha256'], 'config': canonical_sha256(fixture['config'])}
        elif stage in ('conditioning', 'initial'):
            inputs = {'reference_input': expected[name]['sha256'], 'config': canonical_sha256(fixture['config'])}
        elif stage == 'dit':
            dependencies = ['initial' if step == 0 else f'step-{step-1}', 'padded-text',
                            'constant-0', 'constant-1', 'constant-2']
            inputs = {key: measured[key]['sha256'] for key in dependencies}
            inputs['time_schedule'] = canonical_sha256({'step': step, 'steps': fixture['steps'], 'shift': 4.0})
        elif name.startswith('step-'):
            inputs = {key: measured[key]['sha256'] for key in
                      [f'prediction-{step}', 'initial' if step == 0 else f'step-{step-1}']}
        else:
            predecessor = {'final': f'step-{fixture["steps"]-1}', 'unpacked': 'final', 'decoded': 'unpacked'}[name]
            inputs = {predecessor: measured[predecessor]['sha256']}
        rows.append({'boundary': name, 'step': step, 'stage': stage,
                     'precision': result['dit_precision'] if stage == 'dit' else 'fp32',
                     'input_sha256': canonical_sha256(inputs), 'input_identities': inputs,
                     'reference_sha256': expected[name]['sha256'], 'candidate_sha256': m['sha256'],
                     'metrics': {key: m[key] for key in ('nrmse', 'max_abs_error', 'reference_max_abs')},
                     'passed': m['passed']})
    validate_rows(rows)
    return rows


def summarize_run(directory):
    directory = Path(directory).resolve()
    verified = audit(directory)  # Recomputes tensor/pixel metrics and all saved identities.
    result = json.loads((directory/'result.json').read_text())
    fixture = json.loads((directory/'reference/fixture.json').read_text())
    verify_candidate_snapshot(directory,result,fixture)
    rows = build_rows(result, fixture)
    return {'run': str(directory), 'scope': verified['scope'],
            'conditioning_source': verified['conditioning_source'],
            'trajectory_mode': 'free_running', 'native_acceptance_eligible': verified['conditioning_source'] == 'native_text_encoder',
            'diagnostic_embeddings': result.get('diagnostic_embeddings'),
            'reference_fixture_sha256': result['reference_fixture_sha256'],
            'runner_sha256': result['runner_sha256'], 'package_manifest_sha256': result['package_manifest_sha256'],
            'result_sha256': hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),
            'first_failed_boundary': first_failed_boundary(rows),
            'failed_boundaries': [row['boundary'] for row in rows if not row['passed']],
            'passed_tensor_count': sum(row['passed'] for row in rows), 'tensor_count': len(rows),
            'png': verified['png'], 'passed': verified['passed'], 'rows': rows,
            'interpretation': 'Boundary changes locate sensitivity; error differences are not additive causal percentages.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output file to preserve earlier diagnostics')
    records = [summarize_run(path) for path in args.run]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 1, 'scope': 'Audited saved development trajectories; no formal corpus evaluation',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'runs': records}
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    for record in records:
        print(json.dumps({key: record[key] for key in ('run', 'first_failed_boundary', 'passed_tensor_count',
                                                      'tensor_count', 'native_acceptance_eligible')}))


if __name__ == '__main__':
    main()
