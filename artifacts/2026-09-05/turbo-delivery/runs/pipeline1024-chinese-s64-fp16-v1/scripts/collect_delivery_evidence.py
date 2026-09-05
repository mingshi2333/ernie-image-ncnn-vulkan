#!/usr/bin/env python3
"""Freeze Turbo delivery results while retaining failed precision gates."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
from package_model import ROOT, sha256

CASES = {
    'apple-fp16': ('pipeline1024-portable-direct-v1', True),
    'long-fp16': ('pipeline1024-long-s64-v1', False),
    'long-fp32': ('pipeline1024-long-s64-fp32-v1', True),
    'chinese-fp32': ('pipeline1024-chinese-s64-fp32-v1', True),
}
COMPONENTS = ['pipeline64-cuda-reference-v1', 'vae-direct-64-v1',
              'vae-direct-1024-v1', 'text-s64-real-long-v1']


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
             'scope': 'Three fixed 1024 prompts, with the rejected FP16 long-prompt run retained; no broad perceptual quality claim',
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
