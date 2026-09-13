#!/usr/bin/env python3
"""Package a tested native CI build with its launcher and runtime libraries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET

from build_release import collect_notices


def command(*args):
    return subprocess.check_output([str(a) for a in args], text=True).strip()


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'size': path.stat().st_size, 'sha256': digest.hexdigest()}


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    target.chmod(target.stat().st_mode | 0o200)


def target_crates(source):
    host = re.search(r'^host: (.+)$', command('rustc', '-vV'), re.M).group(1)
    metadata = json.loads(command('cargo', 'metadata', '--manifest-path', source / 'tokenizer/Cargo.toml',
                                  '--locked', '--offline', '--format-version', '1', '--filter-platform', host))
    nodes = {node['id']: node for node in metadata['resolve']['nodes']}
    pending, reached = [metadata['resolve']['root']], set()
    while pending:
        key = pending.pop()
        if key not in reached:
            reached.add(key)
            pending.extend(nodes[key]['dependencies'])
    return host, {p['name'] + '-' + p['version'] for p in metadata['packages'] if p['id'] in reached}


def linux_libraries(binary, root):
    listing = command('ldd', binary)
    if 'not found' in listing:
        raise ValueError(listing)
    # glibc and the C++ ABI remain the distribution's system runtime. Package
    # PNG, zlib and OpenMP so users do not need development packages.
    external = {'libc.so.6', 'libm.so.6', 'libmvec.so.1', 'libdl.so.2', 'libpthread.so.0',
                'librt.so.1', 'libstdc++.so.6', 'libgcc_s.so.1'}
    bundled = []
    for name, path in re.findall(r'^\s*(\S+) => (/\S+) ', listing, re.M):
        if name in external:
            continue
        if not name.startswith(('libpng', 'libz.so', 'libgomp.so', 'libomp.so')):
            raise ValueError('Unreviewed non-system runtime library: ' + name)
        copy(Path(path), root / 'lib' / name)
        bundled.append(name)
    # Debian package copyright files cover bundled libraries and the compiler
    # runtime exceptions. Keep all relevant installed package notices.
    for pattern in ('libpng*', 'zlib1g', 'libgomp*', 'gcc-*-base', 'libstdc++*', 'libgcc*'):
        for path in Path('/usr/share/doc').glob(pattern + '/copyright'):
            copy(path, root / 'licenses/system' / path.parent.name / 'copyright')
    for path in Path('/usr/share/common-licenses').glob('*'):
        if path.is_file():
            copy(path, root / 'licenses/system/common' / path.name)
    return {'bundled': bundled, 'system': sorted(external), 'observed': listing,
            'minimum': 'Ubuntu 24.04 x86_64 or compatible glibc 2.39 / libstdc++ from GCC 13'}


def mac_libraries(binary, root):
    prefixes = {name: Path(command('brew', '--prefix', name))
                for name in ('libpng', 'vulkan-loader', 'molten-vk')}
    molten = prefixes['molten-vk'] / 'lib/libMoltenVK.dylib'
    queue = [(binary, root / 'bin/ernie-image'), (molten, root / 'lib/libMoltenVK.dylib')]
    seen = set()
    for original, target in queue:
        if original.resolve() in seen:
            continue
        seen.add(original.resolve())
        if original != target:
            copy(original, target)
        changes = []
        for line in command('otool', '-L', original).splitlines()[1:]:
            name = line.strip().split(' (', 1)[0]
            if name.startswith(('/usr/lib/', '/System/Library/')):
                continue
            if target.suffix == '.dylib' and Path(name).name == original.name:
                continue  # dylib's own LC_ID_DYLIB
            dep = Path(name)
            if name.startswith('@'):
                candidates = [original.parent / Path(name).name]
                candidates += [p / 'lib' / Path(name).name for p in prefixes.values()]
                dep = next((p for p in candidates if p.is_file()), dep)
            if not dep.is_file():
                raise ValueError('Unresolved Mach-O dependency: ' + name)
            dest = root / 'lib' / dep.name
            queue.append((dep, dest))
            relative = '@loader_path/' + ('../lib/' if target.parent.name == 'bin' else '') + dep.name
            changes += ['-change', name, relative]
        if target.suffix == '.dylib':
            changes += ['-id', '@rpath/' + target.name]
        if changes:
            command('install_name_tool', *changes, target)
    for target in [*sorted((root / 'lib').glob('*')), root / 'bin/ernie-image']:
        command('codesign', '--force', '--sign', '-', target)
        linked = command('otool', '-L', target)
        if '/opt/homebrew/' in linked or '/usr/local/' in linked:
            raise ValueError('Unrelocated Homebrew dependency: ' + linked)
    driver = json.loads((prefixes['molten-vk'] / 'etc/vulkan/icd.d/MoltenVK_icd.json').read_text())
    driver['ICD']['library_path'] = '../lib/libMoltenVK.dylib'
    (root / 'vulkan').mkdir()
    (root / 'vulkan/MoltenVK_icd.json').write_text(json.dumps(driver, indent=2) + '\n')
    for name, prefix in prefixes.items():
        for path in prefix.resolve().rglob('*'):
            if path.is_file() and re.match(r'(?i)^(license|copying|notice|copyright)', path.name):
                copy(path, root / 'licenses/system' / name / path.relative_to(prefix.resolve()))
    return {'bundled': sorted(p.name for p in (root / 'lib').iterdir()),
            'minimum': 'macOS 15, ' + platform.machine(), 'codesign': 'ad-hoc; not notarized'}


def windows_libraries(binary, root):
    redist = Path(os.environ['VCToolsRedistDir'])
    dlls = ('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'vcomp140.dll')
    copied = []
    pending = list(dlls)
    for name in pending:
        if name.lower() in copied:
            continue
        matches = sorted((redist / 'x64').glob('Microsoft.VC*.*/' + name))
        if len(matches) != 1:
            raise ValueError('Expected one MSVC x64 redistributable: ' + name + ' in ' + str(redist))
        copy(matches[0], root / 'bin' / name)
        copied.append(name.lower())
        for dep in re.findall(r'^\s+([\w.-]+\.dll)\s*$', command('dumpbin', '/dependents', matches[0]), re.M | re.I):
            if dep.lower().startswith(('msvcp', 'vcruntime', 'concrt', 'vcomp')):
                pending.append(dep.lower())
    vs = Path(os.environ['VSINSTALLDIR'])
    notices = []
    for folder in (vs / 'Licenses', vs / 'Common7/IDE/1033', redist):
        if not folder.is_dir():
            continue
        for path in folder.rglob('*'):
            if path.is_file() and re.match(r'(?i)^(license|redist).*(txt|rtf|html)$', path.name):
                if path.stat().st_size < 2 * 1024 * 1024:
                    copy(path, root / 'licenses/system/msvc' / path.relative_to(vs))
                    notices.append(path.relative_to(vs).as_posix())
    if not notices:
        raise ValueError('MSVC redistribution terms were not found in the Visual Studio installation')
    vcpkg = Path(os.environ['VCPKG_INSTALLATION_ROOT']) / 'installed/x64-windows-static-md/share'
    for name in ('libpng', 'zlib'):
        copy(vcpkg / name / 'copyright', root / 'licenses/system' / name / 'copyright')
    return {'bundled': copied, 'minimum': 'Windows 10/11 x64; system Universal CRT',
            'notices': notices, 'dependencies': command('dumpbin', '/dependents', binary)}


def package(source, binary, output, junit):
    cases = list(ET.parse(junit).getroot().iter('testcase'))
    if not cases or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError('A completed, successful CTest record is required')
    system = platform.system()
    arch = {'AMD64': 'x86_64', 'aarch64': 'arm64'}.get(platform.machine(), platform.machine())
    label = {'Linux': 'linux', 'Windows': 'windows', 'Darwin': 'macos'}[system] + '-' + arch
    name = 'ernie-image-' + label
    output.mkdir(parents=True, exist_ok=True)
    root = output / name
    root.mkdir()  # Never merge into an old payload.
    executable = root / 'bin' / binary.name
    copy(binary, executable)
    dependencies = {'Linux': linux_libraries, 'Windows': windows_libraries, 'Darwin': mac_libraries}[system](binary, root)
    copy(source / 'tools/runtime/run.py', root / 'run.py')
    for path in ('download_model.py', 'release_manifest.py'):
        copy(source / 'tools' / path, root / 'tools' / path)
    for path in ('turbo-v1.json', 'pe-v1.json', 'LICENSE-ERNIE-Image', 'NOTICE'):
        copy(source / 'docs/models' / path, root / 'manifests' / path)
    copy(source / 'LICENSE', root / 'LICENSE')
    copy(source / 'tools/runtime/README.md', root / 'README.md')
    copy(junit, root / 'build-info/ctest-results.xml')
    registry = Path(os.environ.get('CARGO_HOME', str(Path.home() / '.cargo'))) / 'registry/src'
    (root / 'licenses').mkdir(exist_ok=True)
    notices = collect_notices(source, root / 'licenses/source', cargo_registry=registry)
    host, active_crates = target_crates(source)
    inactive = {p['name'] + '-' + p['version'] for p in notices['rust_lock_packages']} - active_crates
    # Cargo.lock also contains firmware/WASI-only packages. Require notices for
    # this native target; keep excluded entries visible instead of treating an
    # unused UEFI dependency as a missing dependency of the desktop executable.
    excluded_gaps = [gap for gap in notices['gaps'] if gap.split(':', 1)[0] in inactive]
    required_gaps = [gap for gap in notices['gaps'] if gap not in excluded_gaps]
    if required_gaps:
        raise ValueError('Source notice inventory is incomplete: ' + repr(required_gaps))
    for path in (source / 'third_party/ncnn/glslang/LICENSES').glob('*.txt'):
        copy(path, root / 'licenses/source/glslang/LICENSES' / path.name)
    # Include nested notices in authenticated, locked crates (e.g. Oniguruma,
    # Unicode tables), in addition to the conservative root-level inventory.
    import tarfile
    for crate in notices['rust_lock_packages']:
        if not crate.get('authenticated_archive'):
            continue
        with tarfile.open(crate['authenticated_archive']) as archive:
            for item in archive:
                parts = Path(item.name).parts
                if item.isfile() and len(parts) > 2 and '..' not in parts and not item.name.startswith('/') and item.size < 2 * 1024 * 1024:
                    if re.match(r'(?i)^(license|copying|notice|copyright)', parts[-1]):
                        target = root / 'licenses/source/rust' / item.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.extractfile(item).read())
    rust_docs = Path(command('rustc', '--print', 'sysroot')) / 'share/doc/rust'
    for path in rust_docs.glob('*'):
        if path.is_file() and re.match(r'^(LICENSE|COPYRIGHT)', path.name):
            copy(path, root / 'licenses/rust-toolchain' / path.name)
    revision = command('git', '-C', source, 'rev-parse', 'HEAD')
    record = {'schema_version': 1, 'platform': label, 'source_revision': revision,
              'ncnn_revision': command('git', '-C', source / 'third_party/ncnn', 'rev-parse', 'HEAD'),
              'ci_run': os.environ.get('GITHUB_RUN_ID'), 'dependencies': dependencies,
              'rustc': command('rustc', '--version'),
              'tests': {'passed': sum(c.find('skipped') is None for c in cases),
                        'skipped': sum(c.find('skipped') is not None for c in cases),
                        'scope': 'Native framework contracts; see ctest-results.xml'},
              'models': 'Separate download; pinned manifests are included'}
    (root / 'build-info/runtime.json').write_text(json.dumps(record, indent=2) + '\n')
    notice_inventory = {key: notices[key] for key in ('entries', 'rust_lock_packages', 'inventory_scope')}
    notice_inventory.update(rust_target=host, target_packages=sorted(active_crates),
                            excluded_target_notice_gaps=excluded_gaps)
    (root / 'licenses/source-inventory.json').write_text(json.dumps(notice_inventory, indent=2) + '\n')
    # Actually relocate before publishing the archive, using a path with spaces.
    with tempfile.TemporaryDirectory(prefix='ernie runtime ') as tmp:
        moved = Path(tmp) / name
        shutil.copytree(root, moved)
        env = os.environ.copy()
        for key in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'VK_DRIVER_FILES', 'VK_ICD_FILENAMES'):
            env.pop(key, None)
        for option in ('--help', '--diagnose'):
            result = subprocess.run([os.sys.executable, str(moved / 'run.py'), option],
                                    env=env, cwd=tmp, capture_output=True, text=True, check=True)
            (root / 'build-info' / (option[2:] + '.txt')).write_text(result.stdout + result.stderr)
    inventory = {p.relative_to(root).as_posix(): identity(p) for p in sorted(root.rglob('*')) if p.is_file()}
    (root / 'files.json').write_text(json.dumps(inventory, indent=2) + '\n')
    archive = output / (name + '.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                zipped.write(path, path.relative_to(output))
    (output / (name + '.sha256')).write_text(identity(archive)['sha256'] + '  ' + archive.name + '\n')
    print(json.dumps({'archive': str(archive), **identity(archive), **record}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--junit', type=Path, required=True)
    args = parser.parse_args()
    package(args.source.resolve(), args.binary.resolve(), args.output.resolve(), args.junit.resolve())
