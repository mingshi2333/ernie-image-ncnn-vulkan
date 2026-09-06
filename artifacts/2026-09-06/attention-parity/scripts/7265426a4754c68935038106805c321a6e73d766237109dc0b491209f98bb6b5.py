#!/usr/bin/env python3
"""Freeze Turbo delivery results while retaining failed precision gates."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
from PIL import Image
from package_model import ROOT, sha256

CASES = {
    'apple-fp16': ('pipeline1024-portable-direct-v1', True),
    'long-fp16': ('pipeline1024-long-s64-v1', False),
    'long-fp32': ('pipeline1024-long-s64-fp32-v1', False),
}
COMPONENTS = ['pipeline64-cuda-reference-v1', 'vae-direct-64-v1',
              'vae-direct-1024-v1', 'text-s64-real-long-v1',
              'time-features-delivery-v1', 'diagnostic-long-step6-fp32-v1']


def recheck_metrics(run, result, fixture, gates):
    """Recompute the frozen verdict from saved bytes using NumPy, not the validator."""
    expected = {**fixture['inputs'], **fixture['final']}
    for i, step in enumerate(fixture['outputs']):
        expected.update({f'{name}-{i}': value for name, value in step.items()})
    if set(expected) != {row['tensor'] for row in result['comparisons']} or len(expected) != len(result['comparisons']):
        raise ValueError('Incomplete comparison inventory')
    verdicts = []
    for row in result['comparisons']:
        name = row['tensor']
        reference = np.fromfile(run/'reference'/expected[name]['file'], '<f4').astype('f8')
        actual = np.fromfile(run/'trace'/(name+'.f32'), '<f4').astype('f8')
        if reference.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
            raise ValueError('Invalid tensor while freezing evidence')
        maximum = float(np.max(abs(actual-reference)))
        nrmse = float(np.sqrt(np.sum((actual-reference)**2)/max(np.sum(reference**2), 1e-30)))
        if not (np.isclose(maximum, row['max_abs_error'], rtol=1e-10, atol=1e-15)
                and np.isclose(nrmse, row['nrmse'], rtol=1e-10, atol=1e-15)):
            raise ValueError(f'Saved metrics disagree with tensors: {name}')
        gate = gates['conditioning' if name in fixture['inputs'] else result['dit_precision']]
        passed = nrmse <= gate['nrmse'] and maximum <= gate['atol']+gate['global_rtol']*np.max(abs(reference))
        if name == 'initial':
            passed = np.array_equal(actual, reference)
        if bool(passed) != row['passed']:
            raise ValueError(f'Saved verdict disagrees with gate: {name}')
        verdicts.append(passed)
    native = np.asarray(Image.open(run/'native.png').convert('RGB'))
    reference = np.asarray(Image.open(run/'reference/reference.png').convert('RGB'))
    delta = abs(native.astype('f8')-reference.astype('f8'))
    if float(delta.mean()) != result['png']['mae'] or float(delta.max()) != result['png']['max_abs']:
        raise ValueError('Saved PNG metrics disagree with images')
    decoded = np.fromfile(run/'trace/decoded.f32', '<f4').reshape(3, *native.shape[:2])
    quantized = (np.clip(decoded/2+.5, 0, 1).transpose(1, 2, 0)*255).round().astype('uint8')
    quantization_exact = np.array_equal(native, quantized)
    gate = gates[result['dit_precision']]
    pixel_pass = delta.mean() <= gate['pixel_mae'] and delta.max() <= gate['pixel_max']
    if (bool(pixel_pass) != result['png']['passed']
            or bool(quantization_exact) != result['native_png_quantization_exact']
            or bool(all(verdicts) and pixel_pass and quantization_exact) != result['passed']):
        raise ValueError('Saved full verdict differs from independently recomputed verdict')


def freeze(output):
    if output.exists():
        raise ValueError('Use a new evidence directory')
    output.mkdir(parents=True)
    index = []

    def copy(source, target):
        if not source.is_file() or source.stat().st_size > 2*1024*1024:
            raise ValueError(f'Missing or unexpectedly large evidence: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    def metadata_tree(source, destination):
        for path in sorted(source.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.log', '.cfg', '.param', '.txt', '.py'):
                copy(path, destination/path.relative_to(source))

    for label, (name, expected_pass) in CASES.items():
        run = ROOT/'outputs'/name
        result = json.loads((run/'result.json').read_text())
        fixture = json.loads((run/'reference/fixture.json').read_text())
        gates = json.loads((run/'gates.json').read_text())
        if result['passed'] != expected_pass or not fixture['complete']:
            raise ValueError(f'Unexpected or incomplete case: {label}')
        if sha256(run/'ernie-image.snapshot') != result['runner_sha256']:
            raise ValueError(f'Runner snapshot differs: {label}')
        validator = run/'scripts/validate_pipeline.py'
        if not validator.exists():
            validator = ROOT/'outputs/validator-source-variants'/(result['validator_sha256']+'.py')
        if sha256(validator) != result['validator_sha256']:
            raise ValueError(f'Validator snapshot differs: {label}')
        for name, digest in result.get('source_snapshot', {}).items():
            if sha256(run/'scripts'/name) != digest:
                raise ValueError(f'Source snapshot differs: {label}/{name}')
        package_path = Path(result['command'][result['command'].index('--model')+1])
        if sha256(package_path/'manifest.json') != result['package_manifest_sha256']:
            raise ValueError(f'Package manifest changed: {label}')
        if (sha256(run/'reference/fixture.json') != result['reference_fixture_sha256']
            or sha256(run/'native.png') != result['png']['sha256']
            or sha256(run/'reference/reference.png') != fixture['reference_png_sha256']):
            raise ValueError(f'Fixture or image changed: {label}')
        for row in result['comparisons']:
            if sha256(run/'trace'/(row['tensor']+'.f32')) != row['sha256']:
                raise ValueError(f'Native tensor changed: {label}/{row["tensor"]}')
        entries = [*fixture['inputs'].values(), *fixture['final'].values()]
        for step in fixture['outputs']:
            entries.extend(step.values())
        for entry in entries:
            if sha256(run/'reference'/entry['file']) != entry['sha256']:
                raise ValueError(f'Reference tensor changed: {label}/{entry["file"]}')
        recheck_metrics(run, result, fixture, gates)
        measured = {row['tensor']: row for row in result['comparisons']}
        log = (run/'native.log').read_text()
        rss = int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', log)[1])
        elapsed = float(re.search(r'Total: ([\d.]+) s', log)[1])
        row = {'case': label, 'run': name, 'passed': result['passed'], 'prompt': fixture['prompt'],
               'tokens': len(fixture['ids']), 'config': fixture['config'], 'steps': fixture['steps'],
               'precision': result['dit_precision'], 'comparison_count': len(result['comparisons']),
               'failed_tensors': [r['tensor'] for r in result['comparisons'] if not r['passed']],
               'final_latent_nrmse': measured['final']['nrmse'], 'decoded_nrmse': measured['decoded']['nrmse'],
               'png_mae': result['png']['mae'], 'png_max_abs': result['png']['max_abs'],
               'max_rss_kib': rss, 'max_rss_gib': rss/1024**2, 'total_seconds': elapsed,
               'native_png_sha256': result['png']['sha256'], 'gates': gates[result['dit_precision']],
               'metrics_independently_recomputed': True,
               'reference_environment': fixture['reference_environment']}
        index.append(row)
        metadata_tree(run, output/'runs'/name)
        # rglob intentionally does not follow the reused reference symlink.
        metadata_tree((run/'reference').resolve(), output/'runs'/name/'reference')

    for name in COMPONENTS:
        metadata_tree(ROOT/'outputs'/name, output/'components'/name)
    for name in ['text-block00-s64-v1', 'text-s64-v2']:
        metadata_tree(ROOT/'models'/name, output/'models'/name)
    for name in ['portable-turbo1024-s32-v1', 'turbo1024-s64-portable']:
        for filename in ['manifest.json', 'model.cfg']:
            copy(ROOT/'models'/name/filename, output/'packages'/name/filename)
    # The incomplete attempt used a directory containing only block zero.
    copy(ROOT/'outputs/rebucket-text-s64-v1.log', output/'failures/rebucket-text-source-path.log')
    for prefix in ['build-delivery', 'ctest-delivery', 'python-delivery', 'install-delivery', 'configure-delivery',
                   'package-turbo1024', 'export-text-s64', 'prepare-pipeline1024-s64', 'rebucket-text-s64-v2']:
        for path in sorted((ROOT/'outputs').glob(prefix+'*.log')):
            copy(path, output/'logs'/path.name)
    for name in ['sources.lock.json', 'requirements-reference.lock', 'tokenizer/Cargo.lock', 'tokenizer/Cargo.toml']:
        copy(ROOT/name, output/'environment'/name)
    for path in sorted((ROOT/'outputs/validator-source-variants').glob('*.py')):
        copy(path, output/'validator-source-variants'/path.name)
    (output/'results.json').write_text(json.dumps(index, indent=2, ensure_ascii=False)+'\n')
    files = [ROOT/'CMakeLists.txt', ROOT/'sources.lock.json', ROOT/'.github/workflows/build.yml']
    for directory in ['src', 'tools', 'tests', 'probes', 'tokenizer/src']:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.rs','.py','.param'))
    state = {'created_utc': datetime.now(timezone.utc).isoformat(),
             'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
             'scope': 'Fixed 1024 prompts, with rejected FP16 and FP32 tensor gates retained; no broad perceptual quality claim',
             'runtime_platform': 'Linux, RTX 4060 Laptop 8 GB, Ryzen 7745HX, 32 GB RAM',
             'source_files_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(files))},
             'evidence_sha256': {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(state, indent=2)+'\n')
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output), indent=2, ensure_ascii=False))
