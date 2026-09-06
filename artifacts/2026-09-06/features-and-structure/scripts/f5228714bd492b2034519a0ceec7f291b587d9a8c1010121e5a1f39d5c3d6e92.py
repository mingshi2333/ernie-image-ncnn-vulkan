#!/usr/bin/env python3
"""Collect the dated native pipeline evidence, excluding weights, tensors and images."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from prepare_block import ROOT, sha256

RUNS = [
    'time-features-v1', 'denoise02-s288-v1', 'denoise36-s288-v1',
    'text-selection-v1', 'text-block00-validation-v2', 'text-prompt-apple-v1',
    'text-prompt-chinese-v1', 'text-prompt-empty-v1', 'vae-8x8-validation-v1',
    'dit-input-s4160-validation-v1', 'dit-output-s4160-validation-v1',
    'residual-block14-fix-v1', 'vae-128x128-cpu-validation-v3',
    'pipeline64-apple-fp16-v3', 'pipeline1024-apple-fp16-v1', 'cli-input-contract-v1',
]
FAILURES = [
    'text-block00-validation-v1', 'pipeline64-apple-fp16-v1',
    'pipeline64-apple-fp32-v2', 'pipeline-nan-diagnostic-v1',
    'real-activation-diagnostic-v1', 'vae-128x128-validation-v1',
    'vae-128x128-cpu-validation-v1', 'vae-128x128-cpu-validation-v2',
]
MODELS = [
    'text-block00-s32-v2', 'text-s32-v1', 'text-s32-v2',
    'vae-8x8-v1', 'vae-128x128-specialized-v1', 'vae-128x128-v1',
    'dit-heads-s4160-v1', 'dit-s288-residual-v1', 'dit-s4160-residual-v1',
    'residual-diagnostic-block14-v1',
]
LOG_PREFIXES = (
    'build-denois', 'build-text', 'build-vae', 'build-fp32-residual',
    'build-residual-tests', 'build-pipeline', 'build-cpu-input',
    'build-cpu-final-pipeline', 'build-final-pipeline',
    'ctest-residual', 'ctest-cpu-generator', 'contracts-pipeline',
    'export-text', 'export-vae', 'fetch-pipeline', 'fetch-text',
    'pipeline64-', 'pipeline1024-', 'prepare-pipeline',
    'rebucket-dit', 'specialize-vae', 'text-', 'vae-', 'vae128-',
    'denoise', 'time-features', 'python-compile-pipeline',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error('Use a new evidence directory')
    out.mkdir(parents=True)

    def copy(source, target):
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f'Evidence unexpectedly large: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    def tree(source, target):
        if not source.is_dir():
            raise FileNotFoundError(source)
        for path in sorted(source.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.log', '.param', '.cfg', '.txt'):
                copy(path, target / path.relative_to(source))

    for category, names in [('runs', RUNS), ('failures', FAILURES)]:
        for name in names:
            tree(ROOT / 'outputs' / name, out / category / name)
    for name in MODELS:
        tree(ROOT / 'models' / name, out / 'models' / name)
    for name in ('pipeline64-v1', 'pipeline64-residual-v1', 'pipeline1024-residual-v1'):
        for file in ('manifest.json', 'model.cfg'):
            copy(ROOT / 'models' / name / file, out / 'packages' / name / file)
    copy(ROOT / 'models/tokenizer/manifest.json', out / 'models/tokenizer/manifest.json')
    for path in sorted((ROOT / 'models/official').iterdir()):
        if path.name.startswith(('text-', 'text_encoder-', 'vae-')) and path.suffix == '.json':
            copy(path, out / 'official-components' / path.name)
    for path in sorted((ROOT / 'outputs').iterdir()):
        if path.is_file() and path.suffix in ('.log', '.json') and path.name.startswith(LOG_PREFIXES):
            copy(path, out / 'logs' / path.name)
    for file in ('sources.lock.json', 'requirements-reference.lock', 'tokenizer/Cargo.lock', 'tokenizer/Cargo.toml'):
        copy(ROOT / file, out / 'environment' / file)

    from source_inventory import source_files
    paths = source_files(ROOT)
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(paths))}
    (out / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')

    def command(args):
        return subprocess.check_output(args, cwd=ROOT, text=True, timeout=20).strip()

    environment = {
        'compiler': command(['clang++', '--version']).splitlines()[0],
        'cmake': command(['cmake', '--version']).splitlines()[0],
        'rust': command(['rustc', '--version']),
        'gpu': command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
        'cpu': next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')),
        'png': command(['pkg-config', '--modversion', 'libpng']),
    }
    (out / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    small = json.loads((ROOT / 'outputs/pipeline64-apple-fp16-v3/result.json').read_text())
    large = json.loads((ROOT / 'outputs/pipeline1024-apple-fp16-v1/result.json').read_text())
    vae = json.loads((ROOT / 'outputs/vae-128x128-cpu-validation-v3/matrix.json').read_text())[0]
    summary = {
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'parent_commit': command(['git', 'rev-parse', 'HEAD']),
        'ncnn_revision': command(['git', '-C', 'third_party/ncnn', 'rev-parse', 'HEAD']),
        'scope': '64x64 full numerical comparison and one 1024x1024 native functional/resource run',
        'full64_passed': small['passed'], 'full64_png': small['png'],
        'full64_final': next(item for item in small['comparisons'] if item['tensor'] == 'final'),
        'vae1024_passed': vae['passed'], 'vae1024_output': vae['outputs']['out0'],
        'full1024_passed': large['passed'], 'full1024_quality_validated': large['quality_validated'],
        'full1024_image': large['image'],
        'full1024_resources': {key: large[key] for key in (
            'max_rss_kib', 'total_seconds', 'text_seconds', 'denoise_seconds', 'vae_and_png_seconds',
            'gpu_device_total_mib', 'system_memory_before_kib', 'system_memory_after_kib')},
        'binaries_sha256': {name: sha256(ROOT / name) for name in ('build/ernie-image', 'build-cpu/ernie-image')},
        'source_capture_scope': 'Current source snapshot; individual results retain their historical binary/tool hashes. '
                                'Later CLI input finite checks and benchmark error handling do not retroactively change runs.',
        'retained_failures': FAILURES,
        'excluded': ['model weights', 'reference/actual tensor payloads', 'binaries', 'generated images'],
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'files': sum(p.is_file() for p in out.rglob('*')),
                      'bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                      'full64_passed': small['passed'], 'full1024_passed': large['passed']}))
    # Interpret results and write README before sealing SHA256SUMS.


if __name__ == '__main__':
    main()
