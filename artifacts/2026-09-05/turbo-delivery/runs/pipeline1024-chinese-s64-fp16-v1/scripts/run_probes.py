#!/usr/bin/env python3
"""Run the small native-cache probes and retain exit codes and build provenance."""
import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def capture(command, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=15)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=Path('build'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--backend', choices=('both', 'cpu', 'vulkan'), default='both')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve()
    out = args.output_dir.resolve()
    binary = build / ('ernie-attention-probe.exe' if platform.system() == 'Windows' else 'ernie-attention-probe')
    if not binary.is_file():
        parser.error(f'Build the probe first; executable not found: {binary}')
    if out.exists() and any(out.glob('*.json*')):
        parser.error('Output already contains results; choose a new directory to preserve prior evidence')
    out.mkdir(parents=True, exist_ok=True)
    sources = ['CMakeLists.txt', 'sources.lock.json', 'probes/attention_probe.cpp', 'tools/run_probes.py']
    manifest = {
        'schema_version': 1,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(),
        'ncnn_revision': capture(['git', '-C', str(root / 'third_party/ncnn'), 'rev-parse', 'HEAD']),
        'project_revision': capture(['git', 'rev-parse', 'HEAD'], root),
        'source_sha256': {name: digest(root / name) for name in sources},
        'executable_sha256': digest(binary),
        'gpu_inventory': capture(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader']),
        'results': [],
        'scope': 'Synthetic causal GQA operator validation, not ERNIE model parity or a speed benchmark',
    }
    cache = build / 'CMakeCache.txt'
    if cache.is_file():
        prefixes = ('CMAKE_BUILD_TYPE:', 'CMAKE_CXX_COMPILER:', 'CMAKE_CXX_FLAGS_RELEASE:', 'NCNN_', 'ERNIE_')
        manifest['cmake_settings'] = [line for line in cache.read_text().splitlines() if line.startswith(prefixes)]
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        manifest['cpu_model'] = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                                      if line.startswith('model name')), None)
    failed = False
    backends = ('cpu', 'vulkan') if args.backend == 'both' else (args.backend,)
    for backend in backends:
        command = [str(binary), '--backend', backend]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            stdout, stderr, code = result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or b''
            stderr = error.stderr or b''
            stdout = stdout.decode(errors='replace') if isinstance(stdout, bytes) else stdout
            stderr = stderr.decode(errors='replace') if isinstance(stderr, bytes) else stderr
            stderr += '\nProbe timed out after 90 seconds\n'
            code = 124
        (out / f'{backend}.jsonl').write_text(stdout)
        (out / f'{backend}.stderr.log').write_text(stderr)
        try:
            records = [json.loads(line) for line in stdout.splitlines() if line.strip()]
            cases = [row for row in records if 'passed' in row]
            coherent = bool(cases) and all(row['passed'] for row in cases)
        except (json.JSONDecodeError, TypeError, KeyError):
            cases, coherent = [], False
        status = 'skipped' if code == 77 else 'passed' if code == 0 and coherent else 'failed'
        failed |= status == 'failed'
        entry = {'backend': backend, 'exit_code': code, 'status': status, 'passed_cases': sum(row.get('passed') is True for row in cases),
                 'stdout_sha256': digest(out / f'{backend}.jsonl'), 'stderr_sha256': digest(out / f'{backend}.stderr.log')}
        manifest['results'].append(entry)
        print(json.dumps(entry, ensure_ascii=False))
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
