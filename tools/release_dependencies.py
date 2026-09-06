#!/usr/bin/env python3
"""Collect Linux build/archive/loader dependency evidence; never approve release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import shlex
import subprocess
import tarfile
import tomllib


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True,
                          env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C',
                               'HOME': str(Path.home())}).stdout


def closure(metadata):
    nodes = {n['id']: n for n in metadata['resolve']['nodes']}
    packages = {p['id']: p for p in metadata['packages']}
    root = metadata['resolve']['root']
    pending = [root]
    seen = set()
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        if key not in nodes or key not in packages:
            raise ValueError('Incomplete resolved dependency graph')
        seen.add(key)
        for dep in nodes[key]['deps']:
            if any(k['kind'] != 'dev' for k in dep['dep_kinds']):
                pending.append(dep['pkg'])
    return [dict(packages[k], enabled_features=nodes[k]['features']) for k in sorted(seen)]


def elf_identity(path):
    with Path(path).open('rb') as stream:
        header = stream.read(20)
    if len(header) != 20 or header[:4] != b'\x7fELF' or header[4] not in (1, 2) or header[5] not in (1, 2):
        raise ValueError('Invalid ELF header')
    return {'class': 32 if header[4] == 1 else 64, 'endian': header[5],
            'machine': struct.unpack('<H' if header[5] == 1 else '>H', header[18:20])[0]}


def loader_paths(text):
    paths = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('linux-vdso.so.'):
            continue
        if '=> not found' in line:
            raise ValueError('Unresolved dynamic dependency')
        match = re.fullmatch(r'(?:\S+ => )?(/.+?) \(0x[0-9a-fA-F]+\)', line)
        if not match:
            raise ValueError('Unrecognized loader output: ' + line)
        paths.append(Path(match[1]))
    if not paths:
        raise ValueError('Empty loader resolution')
    return sorted(set(paths))


def collect_loader(binary, output):
    identity = elf_identity(binary)
    if identity != {'class': 64, 'endian': 1, 'machine': 62}:
        raise ValueError('Only Linux x86_64 evidence is supported')
    headers = run(['readelf', '-l', str(binary)])
    match = re.search(r'Requesting program interpreter: ([^\]]+)\]', headers)
    if not match or match[1] != '/lib64/ld-linux-x86-64.so.2':
        raise ValueError('Unsupported ELF interpreter')
    interpreter = Path(match[1])
    if elf_identity(interpreter) != identity:
        raise ValueError('Interpreter architecture mismatch')
    command = [str(interpreter), '--list', str(binary.resolve())]
    text = run(command)  # glibc loader listing does not enter the program main.
    (output / 'loader.txt').write_text(text)
    libraries = []
    for path in loader_paths(text):
        if elf_identity(path) != identity:
            raise ValueError('Resolved library architecture mismatch')
        owner = run(['rpm', '-qf', '--qf', '%{NEVRA}\n%{SOURCERPM}\n', str(path.resolve())]).strip()
        libraries.append({'path': str(path), 'resolved_path': str(path.resolve()),
                          'sha256': sha(path), 'elf': identity, 'rpm_owner_and_source': owner})
    return {'binary_sha256': sha(binary), 'elf': identity, 'command': command,
            'environment': 'PATH=/usr/bin:/bin LC_ALL=C HOME=current; no LD_* variables',
            'libraries': libraries, 'scope': 'Current host loader resolution; not bundled; Vulkan ICD dlopen is outside --list'}


def crate_notices(package, lock, cache, output):
    name, version = package['name'], package['version']
    entry = next((x for x in lock['package'] if (x['name'], x['version']) == (name, version)), None)
    if not entry or not entry.get('checksum'):
        return {'status': 'local_project_or_unverified', 'notices': []}
    matches = list(Path(cache).glob(f'*/{name}-{version}.crate'))
    matches = [p for p in matches if sha(p) == entry['checksum']]
    if not matches:
        raise ValueError(f'Missing authenticated crate archive: {name} {version}')
    archive = matches[0]
    notices = []
    prefix = f'{name}-{version}/'
    with tarfile.open(archive) as tar:
        for member in tar:
            relative = Path(member.name)
            if member.name.startswith('/') or '..' in relative.parts or not member.name.startswith(prefix):
                raise ValueError('Unsafe crate member')
            if not member.isfile():
                continue
            basename = relative.name.upper()
            if not (basename.startswith(('LICENSE', 'LICENCE', 'COPYING', 'NOTICE', 'COPYRIGHT'))):
                continue
            if member.size > 4 * 1024 * 1024:
                raise ValueError('Oversized notice')
            data = tar.extractfile(member).read()
            destination = output / 'notices' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            notices.append({'path': str(destination.relative_to(output)), 'sha256': sha(destination),
                            'archive_member': member.name})
    return {'status': 'authenticated_texts' if notices else 'missing_text',
            'archive_sha256': entry['checksum'],
            'url': f'https://static.crates.io/crates/{name}/{name}-{version}.crate', 'notices': notices}


def archive_crates(archive, deps, packages):
    members = run(['ar', 't', str(archive)]).splitlines()
    pairs = sorted(set((m[1], m[2]) for name in members
                       if (m := re.match(r'^([A-Za-z0-9_]+)-([0-9a-f]{16})\.', name))))
    matched = []
    unassigned = []
    for name, artifact in pairs:
        dep = deps / f'{name}-{artifact}.d'
        candidates = []
        if dep.is_file():
            text = dep.read_text()
            for p in packages:
                if p['name'].replace('-', '_') == name and (str(Path(p['manifest_path']).parent) + '/' in text or any(Path(t.rstrip(':')).resolve() == Path(p['manifest_path']).parent / 'src/lib.rs' for t in text.split() if t.startswith('/'))):
                    candidates.append(p)
        if len(candidates) != 1:
            unassigned.append({'crate': name, 'artifact': artifact,
                               'reason': 'No unique absolute project dep-info association; see runtime archive evidence'})
        else:
            p = candidates[0]
            matched.append({'id': p['id'], 'name': p['name'], 'version': p['version'],
                            'artifact': artifact, 'dep_info_sha256': sha(dep)})
    native = [m for m in members if not re.match(r'^[A-Za-z0-9_]+-[0-9a-f]{16}\.', m)]
    return {'sha256': sha(archive), 'member_count': len(members), 'matched_project_crates': matched,
            'unassigned_crates': unassigned, 'native_object_members': native,
            'scope': 'Archive membership is an upper bound; final linker garbage collection may discard sections'}


def member_hash(archive, member):
    process = subprocess.Popen(['ar', 'p', str(archive), member], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)
    digest = hashlib.sha256()
    with process.stdout:
        while chunk := process.stdout.read(1024 * 1024):
            digest.update(chunk)
    if process.wait() != 0:
        raise ValueError('Cannot read archive member')
    return digest.hexdigest()


def associate_runtime(archive, build, output):
    members = set(run(['ar', 't', str(archive)]).splitlines())
    sources = list(Path('/usr/lib/rustlib/x86_64-unknown-linux-gnu/lib').glob('*.rlib'))
    sources += list((build / 'cargo/release/build').glob('*/out/*.a'))
    associations = []
    for source in sorted(sources):
        common = sorted(members & set(run(['ar', 't', str(source)]).splitlines()))
        common = [m for m in common if m.endswith(('.o', '.obj'))]
        if not common:
            continue
        hashes = {}
        for member in common:
            expected = member_hash(source, member)
            if member_hash(archive, member) != expected:
                raise ValueError('Same-named runtime archive member has different bytes')
            hashes[member] = expected
        owner = (run(['rpm', '-qf', '--qf', '%{NEVRA}\n%{SOURCERPM}\n', str(source)]).strip()
                 if str(source).startswith('/usr/lib/') else None)
        associations.append({'source_archive': str(source), 'sha256': sha(source),
                             'rpm_owner_and_source': owner, 'identical_members': hashes})
    notices = []
    for name in ['/usr/share/licenses/rust/COPYRIGHT', '/usr/share/licenses/rust/LICENSE-APACHE',
                 '/usr/share/licenses/rust/LICENSE-MIT', '/usr/share/licenses/rust-std-static/cargo-vendor.txt',
                 '/usr/share/doc/rust/COPYRIGHT-library.html']:
        source = Path(name)
        if not source.is_file():
            continue
        destination = output / 'notices/toolchain' / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        notices.append({'source': name, 'path': str(destination.relative_to(output)), 'sha256': sha(source),
                        'rpm_owner_and_source': run(['rpm', '-qf', '--qf', '%{NEVRA}\n%{SOURCERPM}\n', name]).strip()})
    return {'archive_associations': associations, 'toolchain_notices': notices,
            'scope': 'Exact member bytes tied to installed runtime archives; vendor license terms still need review'}


def collect(source, build, binary, output, cache):
    output.mkdir(parents=True, exist_ok=False)
    manifest = source / 'tokenizer/Cargo.toml'
    lock_path = manifest.with_name('Cargo.lock')
    locked = sha(lock_path)
    rust_info_path = build / 'cargo/.rustc_info.json'
    rust_info = json.loads(rust_info_path.read_text())
    compiler_records = [v['stdout'] for v in rust_info['outputs'].values()
                        if v.get('stdout', '').startswith('rustc ')]
    if len(compiler_records) != 1 or '\nhost: x86_64-unknown-linux-gnu\n' not in compiler_records[0]:
        raise ValueError('Missing unique historical Rust host evidence')
    # This build used no --target; bind its generated command and compiler host.
    commands = list(build.glob('tokenizer/CMakeFiles/*/build.make'))
    if (build / 'build.ninja').is_file():
        commands.append(build / 'build.ninja')
    matches = []
    for command_file in commands:
        for line in command_file.read_text().splitlines():
            if 'cargo build --release --locked' not in line:
                continue
            tokens = shlex.split(line)
            if '--manifest-path' not in tokens:
                continue
            candidate = Path(tokens[tokens.index('--manifest-path') + 1])
            if candidate.resolve() == manifest.resolve():
                if '--target' in tokens or 'CARGO_BUILD_TARGET=' in line:
                    raise ValueError('Explicit target build requires separate contract')
                cargo_index = next(i for i, t in enumerate(tokens) if Path(t).name == 'cargo')
                expected = ['build', '--release', '--locked', '--manifest-path', str(candidate),
                            '--target-dir', tokens[tokens.index('--target-dir') + 1]]
                if tokens[cargo_index + 1:] != expected:
                    raise ValueError('Unsupported Cargo build feature/profile options')
                matches.append((command_file, line))
    if len(matches) != 1:
        raise ValueError('Expected one implicit-host locked release command')
    command_files = [matches[0][0]]
    command = ['cargo', 'metadata', '--manifest-path', str(manifest.resolve()), '--locked', '--offline',
               '--filter-platform', 'x86_64-unknown-linux-gnu', '--format-version', '1']
    raw = run(command)
    if sha(lock_path) != locked:
        raise ValueError('Cargo.lock changed during inspection')
    (output / 'cargo-metadata.json').write_text(raw)
    metadata = json.loads(raw)
    packages = closure(metadata)
    lock = tomllib.loads(lock_path.read_text())
    enabled = {(p['name'], p['version']) for p in packages}
    excluded = [p for p in lock['package'] if (p['name'], p['version']) not in enabled]
    archive = archive_crates(build / 'cargo/release/libernie_tokenizer_bridge.a', build / 'cargo/release/deps', packages)
    notes = {p['id']: crate_notices(p, lock, cache, output) for p in packages}
    result = {'schema_version': 1, 'distributable': False, 'licenses_complete': False,
              'scope': 'Linux target dependency evidence; not redistribution approval',
              'cargo_lock_sha256': locked, 'cargo_metadata_command': command,
              'cargo_metadata_sha256': sha(output / 'cargo-metadata.json'),
              'historical_rust_info_sha256': sha(rust_info_path), 'historical_rustc': compiler_records[0],
              'build_command_file': str(command_files[0]), 'build_command_sha256': sha(command_files[0]), 'build_command': matches[0][1],
              'packages': [{'id': p['id'], 'name': p['name'], 'version': p['version'],
                            'enabled_features': p['enabled_features'], 'license_expression': p['license']} for p in packages],
              'excluded_lock_packages': excluded, 'rust_archive': archive, 'notices': notes,
              'loader': collect_loader(binary, output),
              'runtime': associate_runtime(build / 'cargo/release/libernie_tokenizer_bridge.a', build, output),
              'blockers': ['Rust stdlib/compiler native objects require exact vendor notice association',
                           'Archive inclusion does not establish final linked section closure',
                           'System libraries and Vulkan loader/ICD remain external runtime prerequisites',
                           'No redistribution or additional platform approval']}
    (output / 'dependencies.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'build', 'binary', 'output', 'cargo-cache'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    collect(args.source.resolve(), args.build.resolve(), args.binary.resolve(), args.output,
            args.cargo_cache)


if __name__ == '__main__':
    main()
