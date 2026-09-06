#!/usr/bin/env python3
"""Audit saved pipeline runs and freeze their small evidence without changing verdicts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from collect_delivery_evidence import recheck_metrics
from package_model import ROOT, sha256
from pipeline_reference import full_reference_contract, reviewed_shared_reference

# These are the existing, pre-experiment pipeline gates, not fitted to a run.
GATES = {
    'fp32': dict(nrmse=.003, global_rtol=.01, atol=.0002, pixel_mae=.1, pixel_max=2),
    'fp16': dict(nrmse=.15, global_rtol=.25, atol=.03, pixel_mae=12, pixel_max=80),
    'conditioning': dict(nrmse=.0002, global_rtol=.0002, atol=.0002),
}


def check_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'Checksum differs: {path}')


def validator_snapshot(run, result):
    path = run/'scripts/validate_pipeline.py'
    # Early runs recorded the validator digest before per-run script snapshots
    # existed. Never use that format's fallback to hide a damaged modern run.
    if 'source_snapshot' not in result and not path.exists():
        digest = result['validator_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid validator digest')
        path = ROOT/'outputs/validator-source-variants'/(digest+'.py')
    check_hash(path, result['validator_sha256'])
    return path


def audit(run):
    result = json.loads((run/'result.json').read_text())
    fixture = json.loads((run/'reference/fixture.json').read_text())
    gates = json.loads((run/'gates.json').read_text())
    expected_gates=GATES
    if result['dit_precision']=='bf16':
        expected_gates={**GATES, 'bf16':dict(nrmse=.15, global_rtol=.25, atol=.03, pixel_mae=12, pixel_max=80)}
    if not fixture['complete'] or result.get('return_code') != 0 or 'failure' in result or gates != expected_gates:
        raise ValueError(f'Incomplete execution or altered gates: {run}')
    check_hash(run/'ernie-image.snapshot', result['runner_sha256'])
    validator_snapshot(run, result)
    for name, digest in result.get('source_snapshot', {}).items():
        if Path(name).name != name:
            raise ValueError('Invalid script filename')
        check_hash(run/'scripts'/name, digest)
    check_hash(run/'reference/fixture.json', result['reference_fixture_sha256'])
    check_hash(run/'reference/reference.png', fixture['reference_png_sha256'])
    check_hash(run/'native.png', result['png']['sha256'])
    package = Path(result['command'][result['command'].index('--model')+1])
    check_hash(package/'manifest.json', result['package_manifest_sha256'])
    full_reference_contract(fixture, fixture['config'], fixture['prompt'], fixture['steps'])
    model_manifest=json.loads((package/'manifest.json').read_text())
    if model_manifest.get('schema_version')==3:
        if 'package_binding' not in result:
            raise ValueError('Shared result lacks an instance binding')
        from package_dynamic_model import verify_shared_package
        verify_shared_package(package)
        reviewed_shared_reference(run/'reference/fixture.json',result['package_binding'],package/'manifest.json')
    expected_names=set(fixture['inputs'])|set(fixture['final'])
    for i in range(fixture['steps']):expected_names.update((f'prediction-{i}',f'step-{i}'))
    names=[row['tensor'] for row in result['comparisons']]
    if len(names)!=len(expected_names) or set(names)!=expected_names:
        raise ValueError('Result lacks the complete comparison denominator')
    if 'prompt_enhancer' in result:
        pe=result['prompt_enhancer']
        pe_model=Path(result['command'][result['command'].index('--pe-model')+1])
        check_hash(pe_model/'manifest.json',pe['model_manifest_sha256'])
        check_hash(run/'pe-reference/reference.json',pe['reference_manifest_sha256'])
        pe_ref=json.loads((run/'pe-reference/reference.json').read_text())
        for name in ('generated-ids.txt','enhanced.txt'):
            check_hash(run/'pe-reference'/name,pe_ref['files'][name])
        if (not pe.get('tokens_exact') or pe.get('eos')!=pe_ref['eos']
            or (run/'trace/pe-ids.txt').read_bytes()!=(run/'pe-reference/generated-ids.txt').read_bytes()
            or (run/'trace/enhanced-prompt.txt').read_bytes()!=(run/'pe-reference/enhanced.txt').read_bytes()
            or (run/'trace/input-prompt.txt').read_bytes().decode('utf-8')!=pe_ref['input_prompt']):
            raise ValueError('PE trace differs from its official reference')
    entries = [*fixture['inputs'].values(), *fixture['final'].values()]
    for step in fixture['outputs']:
        entries.extend(step.values())
    for entry in entries:
        if Path(entry['file']).name != entry['file']:
            raise ValueError('Invalid reference filename')
        check_hash(run/'reference'/entry['file'], entry['sha256'])
    for row in result['comparisons']:
        if Path(row['tensor']).name != row['tensor']:
            raise ValueError('Invalid tensor name')
        check_hash(run/'trace'/(row['tensor']+'.f32'), row['sha256'])
    if [int(v) for v in (run/'trace/ids.txt').read_text().split()] != fixture['ids']:
        raise ValueError('Tokenizer IDs differ')
    recheck_metrics(run, result, fixture, gates)
    measured = {r['tensor']: r for r in result['comparisons']}
    return dict(run=str(run.resolve()), scope=result['scope'],
                conditioning_source=result.get('conditioning_source', 'native_text_encoder'),
                prompt=fixture['prompt'], tokens=len(fixture['ids']), config=fixture['config'],
                precision=result['dit_precision'], steps=fixture['steps'], passed=result['passed'],
                comparisons=len(measured), passed_comparisons=sum(r['passed'] for r in measured.values()),
                failed_tensors=[r['tensor'] for r in measured.values() if not r['passed']],
                final_latent=measured['final'], decoded=measured['decoded'], png=result['png'],
                resources=result['resources'], runner_sha256=result['runner_sha256'],
                validator_sha256=result['validator_sha256'],
                source_snapshot_coverage=('per_run_scripts' if 'source_snapshot' in result else 'validator_only'),
                independently_recomputed=True)


def freeze(output, runs, evidence, report=None):
    if output.exists() or len({p.name for p in runs}) != len(runs):
        raise ValueError('Require a new output directory and distinct run names')
    # Validate every requested run before creating a partial archive.
    rows = [audit(run) for run in runs]
    output.mkdir(parents=True)

    def copy(source, destination):
        if not source.is_file() or source.stat().st_size > 2*1024*1024:
            raise ValueError(f'Not small evidence: {source}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    for run in runs:
        for name in ('result.json', 'gates.json', 'native.log', 'reference/fixture.json'):
            copy(run/name, output/'runs'/run.name/name)
        result = json.loads((run/'result.json').read_text())
        if 'prompt_enhancer' in result:
            for name in ('pe-reference/reference.json','pe-reference/enhanced.txt','pe-reference/generated-ids.txt',
                         'trace/pe-ids.txt','trace/input-prompt.txt','trace/enhanced-prompt.txt'):
                copy(run/name,output/'runs'/run.name/name)
        validator = validator_snapshot(run, result)
        copy(validator, output/'scripts'/(result['validator_sha256']+'.py'))
        for path in sorted((run/'scripts').glob('*.py')):
            # Content-addressed snapshots avoid duplicating unchanged tools.
            copy(path, output/'scripts'/(sha256(path)+'.py'))
    for path in evidence:
        copy(path, output/'evidence'/path.resolve().relative_to(ROOT))
    if report is not None:
        copy(report, output/'README.md')
    (output/'results.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False)+'\n')
    from source_inventory import source_files
    sources = source_files(ROOT)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    source_files_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(sources))},
                    evidence_sha256={str(p.relative_to(output)): sha256(p)
                                     for p in sorted(output.rglob('*')) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--evidence', action='append', type=Path, default=[])
    parser.add_argument('--report', type=Path, help='Reviewed Markdown report to include in the evidence hashes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output, args.run, args.evidence, args.report), indent=2, ensure_ascii=False))
