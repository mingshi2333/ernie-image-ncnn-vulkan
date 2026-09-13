#!/usr/bin/env python3
"""Download the published model once, then run the bundled native program.

Python is used only for this optional download/launch helper. Inference runs in
bin/ernie-image. All native options are accepted unchanged.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))


def environment(root, system=None):
    env = os.environ.copy()
    system = system or sys.platform
    if system.startswith('linux'):
        env['LD_LIBRARY_PATH'] = str(root / 'lib') + (
            os.pathsep + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    elif system == 'darwin':
        if not env.get('VK_DRIVER_FILES') and not env.get('VK_ICD_FILENAMES'):
            env['VK_DRIVER_FILES'] = str(root / 'vulkan/MoltenVK_icd.json')
    return env


def has_option(args, name):
    return name in args or any(arg.startswith(name + '=') for arg in args)


def prepare(model, manifest, binary, env, pe=False):
    from download_model import download
    from release_manifest import load_manifest
    data = load_manifest(manifest)
    # A quick presence check avoids rehashing 23 GB on every launch. The native
    # loader checks its package contracts; --verify-model requests full hashes.
    complete = all((model / f['path']).is_file() and
                   (model / f['path']).stat().st_size == f['size']
                   for f in data['files'])
    if complete:
        return
    size_gb = sum(f['size'] for f in data['files']) / 1e9
    print(f'Preparing {model.name}: {size_gb:.2f} GB. Interrupted downloads can be resumed.', flush=True)
    download(data, model)
    subprocess.run([str(binary), '--pe-model' if pe else '--model',
                    str(model), '--verify-model'], env=env, check=True)


def main(argv=None, root=ROOT):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--with-pe', action='store_true')
    parser.add_argument('--models-dir', type=Path, default=root / 'models')
    options, args = parser.parse_known_args(argv)
    binary = root / 'bin' / ('ernie-image.exe' if os.name == 'nt' else 'ernie-image')
    if not binary.is_file():
        parser.exit(1, f'Missing {binary}. Extract the complete runtime archive first.\n')
    # Python's zipfile extractor does not preserve executable permission bits.
    if os.name != 'nt' and not os.access(binary, os.X_OK):
        binary.chmod(binary.stat().st_mode | 0o100)
    env = environment(root)
    if not args and not options.download_only:
        args = ['--help']
    if any(a in args for a in ('--help', '-h', '--help-all', '--diagnose')):
        print('Launcher options: --models-dir DIR, --download-only, --with-pe\n'
              'First generation downloads the 23.27 GB main model. PE adds 7.68 GB.\n'
              'Default image size: 512 x 512. Native options follow.\n', flush=True)
        return subprocess.call([str(binary), *args], env=env)
    if not options.download_only and not any(has_option(args, a) for a in
            ('--prompt', '--prompt-file', '--verify-model')):
        parser.exit(2, 'Supply --prompt TEXT, --prompt-file FILE, or --download-only.\n')
    if not has_option(args, '--model'):
        model = options.models_dir.absolute() / 'turbo'
        prepare(model, root / 'manifests/turbo-v1.json', binary, env)
        args += ['--model', str(model)]
    if options.with_pe and not has_option(args, '--pe-model'):
        model = options.models_dir.absolute() / 'pe'
        prepare(model, root / 'manifests/pe-v1.json', binary, env, pe=True)
        args += ['--pe-model', str(model)]
    if options.download_only:
        print('Models are ready. Run again with --prompt to generate an image.')
        return 0
    if not any(has_option(args, a) for a in ('--width', '--height')):
        args += ['--width', '512', '--height', '512']
    if not has_option(args, '--output'):
        args += ['--output', 'output.png']
    return subprocess.call([str(binary), *args], env=env)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'ERNIE startup failed: {error}', file=sys.stderr)
        sys.exit(1)
