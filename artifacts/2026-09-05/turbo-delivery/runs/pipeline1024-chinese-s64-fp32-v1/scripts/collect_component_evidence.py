#!/usr/bin/env python3
"""Copy small local component evidence; never include weights, binaries or tensors."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from prepare_block import ROOT, sha256

RUNS = [
    'latent-ops-v2', 'latent-ops-cpu-only', 'tokenizer-v2', 'block01-s288-direct',
    'sequence01-s288-v2', 'dit-head-input-v2', 'dit-head-output-v2',
    'dit-prefix02-s288-v1', 'dit-prefix02-cpu-only', 'sequence36-s288-v1', 'dit36-s288-v1',
    'reference-block31-calibration', 'reference-block33-calibration',
    'cache-sequence36-s288-v1', 'cache-dit36-s288-v1',
]
FAILURES = ['latent-ops-v1', 'sequence01-s288-v1', 'dit-head-input-v1', 'dit-head-output-v1']
LOGS = [
    'ctest-components.log', 'ctest-heads.log', 'ctest-cpu-heads.log', 'model-contract-download.log',
    'direct-block00.log', 'direct-block00-resources.log', 'tokenizer-v1.log', 'tokenizer-v2.log',
    'rmsnorm-native-fp16-v2.log', 'ctest-rmsnorm-v2.log', 'layernorm-native-fp16.log',
    'fetch-dit-heads.log', 'export-dit-heads-v1.log', 'export-dit-heads-v2.log', 'head-input-gdb.log',
    'full-dit-validation.log', 'full-dit-validation-v2.log', 'full-dit-commands.json', 'dit-s288-conversion.log',
    'dit-s288-conversion-v2.log', 'dit-s288-conversion-v3.log', 'dit-s288-conversion-v4.log',
    'pipeline-cache-comparison.log',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Use a new evidence directory')
    output.mkdir(parents=True)
    missing = []

    def copy(source, destination):
        if not source.is_file():
            missing.append(str(source.relative_to(ROOT)))
            return
        if source.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f'Evidence unexpectedly large: {source}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def tree(source, destination):
        if not source.is_dir():
            missing.append(str(source.relative_to(ROOT)))
            return
        for path in sorted(source.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.log', '.param'):
                copy(path, destination / path.relative_to(source))

    for name in RUNS:
        tree(ROOT / 'outputs' / name, output / 'runs' / name)
    for name in FAILURES:
        tree(ROOT / 'outputs' / name, output / 'failures' / name)
    for name in LOGS:
        copy(ROOT / 'outputs' / name, output / 'logs' / name)
    blocks = []
    for i in range(36):
        batch = f'dit-s288-v{2 if i < 9 else 3 if i < 32 else 4}'
        path = ROOT / f'models/{batch}/block-{i:02d}'
        tree(path, output / f'blocks/block-{i:02d}')
        copy(ROOT / f'models/official/dit-block-{i:02d}.manifest.json',
             output / f'blocks/block-{i:02d}/component.manifest.json')
        matrix = json.loads((path / 'validation/matrix.json').read_text())
        model = json.loads((path / 'runtime/model.json').read_text())
        blocks.append({'block': i, 'batch': batch, 'passed': len(matrix) == 4 and all(x['passed'] for x in matrix),
                       'weights_sha256': model['weights_sha256'],
                       'bytes': (path / 'runtime/block.ncnn.bin').stat().st_size,
                       'configuration_passed': {f"{x['backend']}-{x['precision']}": x['passed'] for x in matrix},
                       'nrmse': {f"{x['backend']}-{x['precision']}": x['nrmse'] for x in matrix}})
    for batch in ('dit-s288-v1', 'dit-s288-v2', 'dit-s288-v3', 'dit-s288-v4'):
        copy(ROOT / f'models/{batch}/conversion-run.json', output / f'batches/{batch}.json')
    tree(ROOT / 'models/dit-s288-v1/block-05/validation', output / 'failures/batch-v1-block05')
    tree(ROOT / 'models/dit-heads-s288-v1/output', output / 'failures/output-head-export-v1')
    tree(ROOT / 'models/dit-heads-s288-v1/input', output / 'heads/input')
    tree(ROOT / 'models/dit-heads-s288-v2/output', output / 'heads/output')
    for path in sorted((ROOT / 'models/official').glob('dit-*.manifest.json')):
        if not path.name.startswith('dit-block-'):
            copy(path, output / 'heads/components' / path.name)
    tree(ROOT / 'models/converted/dit-block-00-s4160/direct-bf16', output / 'direct-block00')
    copy(ROOT / 'models/tokenizer/manifest.json', output / 'tokenizer/manifest.json')
    for name in ('sources.lock.json', 'requirements-reference.lock', 'tokenizer/Cargo.lock', 'tokenizer/Cargo.toml'):
        copy(ROOT / name, output / 'environment' / name)
    original = ROOT / 'models/converted/dit-block-00-s4160/bf16-storage/block.ncnn.bin'
    direct = ROOT / 'models/converted/dit-block-00-s4160/direct-bf16/block.ncnn.bin'
    equality = {'original_sha256': sha256(original), 'direct_sha256': sha256(direct),
                'original_bytes': original.stat().st_size, 'direct_bytes': direct.stat().st_size}
    equality['byte_identical'] = equality['original_sha256'] == equality['direct_sha256']
    (output / 'direct-block00/equality.json').write_text(json.dumps(equality, indent=2) + '\n')
    source_paths = [ROOT / 'CMakeLists.txt', ROOT / '.gitignore', ROOT / 'sources.lock.json']
    for directory in ('src', 'probes', 'tools', 'tests', 'tokenizer'):
        source_paths.extend(path for path in (ROOT / directory).rglob('*')
                            if path.is_file() and '__pycache__' not in path.parts and 'target' not in path.parts
                            and path.suffix in ('.h', '.cpp', '.py', '.rs', '.toml', '.lock'))
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(source_paths))}
    (output / 'source-sha256.json').write_text(json.dumps(source_hashes, indent=2) + '\n')
    def command_text(command):
        try:
            return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
        except (OSError, subprocess.SubprocessError) as error:
            return f'unavailable: {error}'
    environment = {
        'compiler': command_text(['clang++', '--version']).splitlines()[0],
        'cmake': command_text(['cmake', '--version']).splitlines()[0],
        'rust': command_text(['rustc', '--version']),
        'cargo': command_text(['cargo', '--version']),
        'gpu': command_text(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
        'cpu': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                     if line.startswith('model name')), 'unavailable'),
        'memory_total': next((line for line in Path('/proc/meminfo').read_text().splitlines()
                              if line.startswith('MemTotal:')), 'unavailable'),
        'precision_scope': 'CPU FP32; Vulkan FP32/FP16/BF16 storage with FP32 normalization intermediates',
    }
    (output / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    summary = {'captured_at': datetime.now(timezone.utc).isoformat(),
               'parent_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'ncnn_revision': subprocess.check_output(['git', '-C', 'third_party/ncnn', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'scope': 'Component and DiT prediction validation; no complete text-to-image pipeline',
               'blocks': blocks, 'all_blocks_passed': all(item['passed'] for item in blocks),
               'block_weight_bytes': sum(item['bytes'] for item in blocks), 'missing_optional_evidence': missing}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'files': sum(x.is_file() for x in output.rglob('*')),
                      'all_blocks_passed': summary['all_blocks_passed'], 'missing': missing}))
    # README and checksum list are finalized after interpreting the retained results.


if __name__ == '__main__':
    main()
