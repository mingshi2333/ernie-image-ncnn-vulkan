#!/usr/bin/env python3
"""Verify a local runtime archive and run one frozen Linux offline smoke case.

The inference process has no Python environment, network or source checkout.
This tool is an external supervisor; success never authorizes redistribution.
Use --prepare to freeze/extract first, then --run in a bounded cgroup.
"""
import argparse
import hashlib
import gzip
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import struct
import subprocess
import sys
import tarfile
import time
import zlib

GIB = 1024 ** 3
MAX_ARCHIVE_BYTES = GIB
MAX_FILE_BYTES = 512 * 1024 ** 2
PROJECT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise ValueError(f'Expected a regular file: {p}')
    return {'path': str(p.resolve()), 'size': p.stat().st_size, 'sha256': digest(p)}


def checked_record(value):
    if set(value) != {'path', 'size', 'sha256'}:
        raise ValueError('File identity requires exactly path, size and sha256')
    if type(value['size']) is not int or value['size'] < 0:
        raise ValueError('Invalid file size')
    if not isinstance(value['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', value['sha256']):
        raise ValueError('Invalid SHA256')
    actual = identity(value['path'])
    if actual['size'] != value['size'] or actual['sha256'] != value['sha256']:
        raise ValueError('File identity changed: ' + value['path'])
    return actual


def member_path(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise ValueError('Invalid archive path')
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ('', '.', '..') for p in value.split('/')) or str(path) != value:
        raise ValueError('Noncanonical archive path')
    return path


def read_inventory(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, list) or not value or len(value) > 10000:
        raise ValueError('Invalid runtime inventory')
    found = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {'path', 'size', 'sha256'}:
            raise ValueError('Invalid inventory record')
        name = str(member_path(item['path']))
        if name in found or type(item['size']) is not int or not 0 <= item['size'] <= MAX_FILE_BYTES:
            raise ValueError('Duplicate inventory path or excessive size')
        if not isinstance(item['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise ValueError('Invalid inventory SHA256')
        found[name] = item
    if sum(f['size'] for f in found.values()) > MAX_ARCHIVE_BYTES:
        raise ValueError('Runtime payload exceeds bounded extraction size')
    for name in found:
        if any(str(p) in found for p in PurePosixPath(name).parents):
            raise ValueError('Inventory file/directory collision')
    if 'bin/ernie-image' not in found or 'release.json' not in found:
        raise ValueError('Linux CLI and release provenance are required')
    return found


def preflight_tar(archive):
    """Bound extension records before tarfile can allocate/process their data.

    The release writer uses regular files and optional PAX path/time metadata.
    Sparse files, links and unfamiliar PAX semantics are outside this format.
    """
    total = 0
    metadata_bytes = 0
    headers = 0
    with gzip.open(archive, 'rb') as stream:
        while True:
            header = stream.read(512)
            total += len(header)
            if len(header) != 512:
                raise ValueError('Truncated raw tar header')
            if header == bytes(512):
                padding = stream.read(512)
                if padding != bytes(512):
                    raise ValueError('Invalid tar end markers')
                padding_bytes = 0
                while block := stream.read(65536):
                    padding_bytes += len(block)
                    if any(block) or padding_bytes > 1024 * 1024:
                        raise ValueError('Unexpected data after tar end')
                return
            member = tarfile.TarInfo.frombuf(header, 'utf-8', 'surrogateescape')
            headers += 1
            if headers > 30000 or member.size < 0:
                raise ValueError('Excessive tar headers or invalid size')
            padded = (member.size + 511) // 512 * 512
            total += padded
            if total > MAX_ARCHIVE_BYTES + 16 * 1024 ** 2:
                raise ValueError('Decompressed tar exceeds bounded size')
            if member.type in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME):
                metadata_bytes += member.size
                if member.size > 65536 or metadata_bytes > 4 * 1024 ** 2:
                    raise ValueError('Tar metadata exceeds bounded size')
                data = stream.read(padded)
                if len(data) != padded:
                    raise ValueError('Truncated tar metadata')
                if member.type != tarfile.GNUTYPE_LONGNAME:
                    data = data[:member.size]
                    pos = 0
                    while pos < len(data):
                        space = data.find(b' ', pos)
                        if space < 0 or not data[pos:space].isdigit():
                            raise ValueError('Invalid PAX record')
                        size = int(data[pos:space])
                        record = data[space+1:pos+size]
                        if size <= space-pos+1 or pos+size > len(data) or not record.endswith(b'\n') or b'=' not in record:
                            raise ValueError('Invalid PAX record length')
                        key = record.split(b'=', 1)[0]
                        if key not in (b'path', b'mtime', b'atime', b'ctime', b'uid', b'gid', b'uname', b'gname'):
                            raise ValueError('Unsupported PAX metadata semantics')
                        pos += size
            else:
                if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or member.size > MAX_FILE_BYTES:
                    raise ValueError('Unsupported tar member type or size')
                remaining = padded
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError('Truncated raw tar payload')
                    remaining -= len(block)


def extract_verified(archive, inventory, destination):
    """Only regular, enumerated files; no tar extraction directives are executed."""
    expected = read_inventory(inventory)
    preflight_tar(archive)
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    seen = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            full = member_path(member.name)
            if len(full.parts) < 2 or full.parts[0] != 'ernie-runtime' or member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
                raise ValueError('Unexpected archive prefix/type')
            name = PurePosixPath(*full.parts[1:]).as_posix()
            if name not in expected or name in seen or member.size != expected[name]['size']:
                raise ValueError('Archive member set/size mismatch')
            seen.add(name)
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with tar.extractfile(member) as source, target.open('xb') as output:
                remaining = member.size
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError('Truncated archive member')
                    output.write(block)
                    h.update(block)
                    remaining -= len(block)
            if h.hexdigest() != expected[name]['sha256']:
                raise ValueError('Archive member checksum mismatch')
            target.chmod(0o755 if name == 'bin/ernie-image' else 0o644)
    if seen != set(expected):
        raise ValueError('Archive missing inventory members')
    provenance = json.loads((destination / 'release.json').read_text())
    selected = provenance.get('selected_install_files')
    if not isinstance(selected, list) or not selected:
        raise ValueError('Release installation provenance is absent')
    for item in selected:
        if expected.get(item.get('path')) != item:
            raise ValueError('Release installation identity differs from inventory')
    return expected


def validate_case(case):
    fields = {'schema_version', 'scope', 'archive', 'inventory', 'model', 'inputs',
              'expected_png', 'request', 'hidden_paths', 'resources'}
    if set(case) != fields or case['schema_version'] != 1 or type(case['schema_version']) is not int:
        raise ValueError('Unknown smoke-case contract')
    if case['scope'] != 'development_linux_offline_delivery':
        raise ValueError('Only a development delivery case is supported')
    r = case['request']
    if set(r) != {'kind', 'width', 'height', 'steps', 'strength', 'threads', 'text_down_vector'}:
        raise ValueError('Unknown generation request fields')
    if r['kind'] not in ('text-to-image', 'img2img') or (r['width'], r['height']) not in ((512, 384), (1024, 1024)):
        raise ValueError('Unreviewed request kind or static shape')
    if any(type(r[k]) is not int for k in ('width', 'height', 'steps', 'threads')):
        raise ValueError('Request integers cannot be booleans')
    if r['steps'] != 8 or r['threads'] != 2 or type(r['text_down_vector']) is not bool:
        raise ValueError('Only frozen eight-step/two-thread FP32 development is supported')
    strength = r['strength']
    if r['kind'] == 'text-to-image':
        if strength is not None:
            raise ValueError('Text-to-image strength must be null')
        inputs = {'prompt', 'latent'}
    else:
        if type(strength) not in (int, float) or strength not in (0, .5, 1):
            raise ValueError('Unreviewed img2img endpoint')
        inputs = {'image'} if strength == 0 else {'image', 'prompt', 'latent'}
        if strength == 0 and r['text_down_vector']:
            raise ValueError('Strength zero has no text reduction')
    if set(case['inputs']) != inputs:
        raise ValueError('Input set does not match the frozen request')
    if not isinstance(case['hidden_paths'], list) or not case['hidden_paths']:
        raise ValueError('Explicit hidden source roots are required')
    for path in case['hidden_paths']:
        if not isinstance(path, str) or not Path(path).is_absolute() or Path(path).resolve() == Path('/'):
            raise ValueError('Invalid hidden source root')
    limits = case['resources']
    if set(limits) != {'memory_max', 'swap_max', 'host_min', 'timeout_seconds'}:
        raise ValueError('Unknown resource controls')
    if any(type(v) is not int for v in limits.values()):
        raise ValueError('Resource values must be integers')
    if not 4 * GIB <= limits['memory_max'] <= 20 * GIB or limits['swap_max'] != 0 or limits['host_min'] < 3 * GIB:
        raise ValueError('Resource budget exceeds the approved scope')
    if not 1 <= limits['timeout_seconds'] <= 1800:
        raise ValueError('Invalid timeout')
    if set(case['model']) != {'path', 'manifest'}:
        raise ValueError('Model path and immutable manifest identity are required')
    model = Path(case['model']['path']).resolve()
    if not model.is_dir() or Path(case['model']['manifest']['path']).resolve() != model / 'manifest.json':
        raise ValueError('Manifest does not belong to the selected model')
    return case


def prepare(case_path, output):
    case = validate_case(json.loads(Path(case_path).read_text()))
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('Output cannot follow symlinks')
    common = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse',
        '--path-format=absolute', '--git-common-dir'], text=True).strip()
    repository = Path(common).parent.resolve()
    roots = [Path(p).resolve() for p in case['hidden_paths']]
    if not any(p == repository or p in repository.parents for p in roots):
        raise ValueError('Hidden roots must cover this entire source repository')
    for hidden in case['hidden_paths']:
        root = Path(hidden).resolve()
        if root == output or root in output.parents:
            raise ValueError('Delivery output must be outside hidden source roots')
    if case['archive']['size'] > MAX_ARCHIVE_BYTES or case['inventory']['size'] > 10 * 1024 ** 2:
        raise ValueError('Archive/inventory input exceeds bounded size')
    for item in [case['archive'], case['inventory'], case['model']['manifest'],
                 case['expected_png'], *case['inputs'].values()]:
        checked_record(item)
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'preparing', 'distributable': False, 'published': False}
    try:
        (output / 'case.json').write_text(json.dumps(case, indent=2, ensure_ascii=False) + '\n')
        shutil.copy2(__file__, output / 'check_release.py')
        shutil.copyfile(case['inventory']['path'], output / 'files.json')
        inventory = extract_verified(case['archive']['path'], case['inventory']['path'], output / '安装 original')
        (output / '安装 original').rename(output / '移动 installation')
        inputs = output / 'inputs'
        inputs.mkdir()
        mapped = {}
        for key, value in {**case['inputs'], 'expected_png': case['expected_png']}.items():
            dest = inputs / {'image': 'input.png', 'prompt': 'prompt.txt', 'latent': 'initial.f32', 'expected_png': 'expected.png'}[key]
            shutil.copyfile(value['path'], dest)
            if identity(dest)['sha256'] != value['sha256']:
                raise ValueError('Input changed during copying')
            mapped[key] = identity(dest)
        (output / '模型 shared').mkdir()
        (output / 'results').mkdir()
        copied = {'case.json': digest(output / 'case.json'), 'check_release.py': digest(output / 'check_release.py'),
                  'files.json': case['inventory']['sha256']}
        for name in inventory:
            copied['移动 installation/' + name] = inventory[name]['sha256']
        for record in mapped.values():
            copied[str(Path(record['path']).relative_to(output))] = record['sha256']
        result.update(status='prepared_not_executed', files=copied, inputs=mapped,
                      case_sha256=digest(output / 'case.json'), payload_files=len(inventory),
                      source_repository=str(repository), hidden_roots=[str(p) for p in roots],
                      archive=checked_record(case['archive']), inventory=checked_record(case['inventory']))
    except BaseException as error:
        result.update(status='preparation_failed', error=str(error))
        raise
    finally:
        (output / 'preparation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Host memory availability is unavailable')


def cgroup(limits):
    relative = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    path = Path('/sys/fs/cgroup') / relative.lstrip('/')
    observed = {name: (path / name).read_text().strip() for name in ('memory.max', 'memory.swap.max', 'cpu.max')}
    if observed['memory.max'] != str(limits['memory_max']) or observed['memory.swap.max'] != '0':
        raise RuntimeError('Run inside the exact frozen memory/swap cgroup')
    return path, observed


def execute(command, path, limits, name, cg, expected=0):
    started = time.monotonic()
    record = {'argv': command, 'exit_code': None, 'failure': None, 'host_min_observed': available(),
              'peak_cgroup_memory_current': 0, 'cgroup_memory_scope': 'entire supervisor scope; sampled every 50 ms'}
    if record['host_min_observed'] < limits['host_min']:
        raise RuntimeError('Host memory floor failed before execution')
    with (path / (name + '.log')).open('xb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                record['host_min_observed'] = min(record['host_min_observed'], available())
                record['peak_cgroup_memory_current'] = max(record['peak_cgroup_memory_current'], int((cg / 'memory.current').read_text()))
                if record['host_min_observed'] < limits['host_min']:
                    raise RuntimeError('Host memory floor crossed')
                if time.monotonic() - started > limits['timeout_seconds']:
                    raise RuntimeError('Frozen timeout exceeded')
                time.sleep(.05)
            record['exit_code'] = process.wait()
            if record['exit_code'] != expected:
                raise RuntimeError(f'{name} returned {record["exit_code"]}; expected {expected}')
        except BaseException as error:
            record['failure'] = str(error)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            record['exit_code'] = process.wait()
            raise
        finally:
            record['wall_seconds'] = time.monotonic() - started
            record['memory_events'] = (cg / 'memory.events').read_text()
            (path / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    return record


def sandbox(output, model, hidden, *, no_gpu=False):
    """Bind the closed model tree before hiding its original checkout path."""
    command = ['bwrap', '--die-with-parent', '--unshare-net', '--ro-bind', '/', '/',
               '--proc', '/proc', '--dev-bind', '/dev', '/dev', '--tmpfs', '/tmp',
               '--ro-bind', str(output), str(output),
               '--bind', str(output / 'results'), str(output / 'results'),
               '--ro-bind', str(model), str(output / '模型 shared')]
    # Bind sources are opened outside the namespace before hiding their roots.
    roots = sorted({Path(p).resolve() for p in hidden}, key=lambda p: len(p.parts))
    hidden_roots = []
    for root in roots:
        if not any(p == root or p in root.parents for p in hidden_roots):
            hidden_roots.append(root)
            command += ['--tmpfs', str(root)]
    command += ['--clearenv', '--setenv', 'PATH', '/usr/bin:/bin', '--setenv', 'LC_ALL', 'C.UTF-8',
                '--setenv', 'OMP_NUM_THREADS', '2', '--setenv', 'OPENBLAS_NUM_THREADS', '2',
                '--chdir', str(output / 'results')]
    if no_gpu:
        command += ['--setenv', 'VK_ICD_FILENAMES', '/no-vulkan-driver.json',
                    '--setenv', 'VK_DRIVER_FILES', '/no-vulkan-driver.json']
    return command + ['--']


def png_container(path, width, height):
    if Path(path).stat().st_size > 64 * 1024 ** 2:
        raise ValueError('Output PNG exceeds bounded size')
    data = Path(path).read_bytes()
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('Invalid output PNG signature/size')
    pos = 8
    kinds = []
    while pos < len(data):
        if len(data) - pos < 12:
            raise ValueError('Truncated PNG chunk')
        size = struct.unpack('>I', data[pos:pos+4])[0]
        kind = data[pos+4:pos+8]
        if size > len(data) - pos - 12:
            raise ValueError('Truncated PNG data')
        payload = data[pos+8:pos+8+size]
        crc = struct.unpack('>I', data[pos+8+size:pos+12+size])[0]
        if zlib.crc32(kind + payload) & 0xffffffff != crc:
            raise ValueError('PNG CRC mismatch')
        if not kinds and (kind != b'IHDR' or payload != struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)):
            raise ValueError('PNG dimensions or RGB8 format mismatch')
        kinds.append(kind)
        pos += size + 12
        if kind == b'IEND':
            if size or pos != len(data):
                raise ValueError('Malformed PNG end')
            break
    if not kinds or kinds[-1] != b'IEND' or b'IDAT' not in kinds:
        raise ValueError('Incomplete PNG container')


def run(output):
    output = Path(output).resolve()
    prep = json.loads((output / 'preparation.json').read_text())
    if prep.get('status') != 'prepared_not_executed' or (output / 'result.json').exists():
        raise ValueError('A new successfully prepared directory is required')
    case = validate_case(json.loads((output / 'case.json').read_text()))
    if digest(output / 'case.json') != prep['case_sha256'] or digest(output / 'files.json') != case['inventory']['sha256']:
        raise ValueError('Prepared case or payload inventory changed')
    expected_files = {'case.json': prep['case_sha256'], 'files.json': case['inventory']['sha256'],
                      'check_release.py': digest(__file__)}
    for name, record in read_inventory(output / 'files.json').items():
        expected_files['移动 installation/' + name] = record['sha256']
    inputs = {}
    for key, value in {**case['inputs'], 'expected_png': case['expected_png']}.items():
        name = {'image': 'input.png', 'prompt': 'prompt.txt', 'latent': 'initial.f32', 'expected_png': 'expected.png'}[key]
        expected = {'path': str(output / 'inputs' / name), 'size': value['size'], 'sha256': value['sha256']}
        if prep['inputs'].get(key) != expected:
            raise ValueError('Prepared input mapping differs from its canonical identity')
        checked_record(expected)
        inputs[key] = expected
        expected_files['inputs/' + name] = value['sha256']
    if set(prep['inputs']) != set(inputs) or prep['files'] != expected_files:
        raise ValueError('Prepared file/input set does not match the frozen case')
    for name, expected in expected_files.items():
        if identity(output / name)['sha256'] != expected:
            raise ValueError('Prepared file changed: ' + name)
    checked_record(case['model']['manifest'])
    if sys.platform != 'linux' or not shutil.which('bwrap'):
        raise RuntimeError('Actual Linux bubblewrap is required')
    cg, controls = cgroup(case['resources'])
    model = Path(case['model']['path']).resolve()
    for path in model.rglob('*'):
        if path.is_symlink():
            raise ValueError('Offline model must have a closed tree without symlinks')
    binary = output / '移动 installation/bin/ernie-image'
    wrapper = sandbox(output, model, case['hidden_paths'])
    results = output / 'results'
    result = {'status': 'running', 'distributable': False, 'published': False,
              'scope': case['scope'], 'preparation_sha256': digest(output / 'preparation.json'),
              'case_sha256': digest(output / 'case.json'), 'resource_controls': controls,
              'commands': {}, 'network_disabled': False, 'source_hidden': False,
              'full_quality_gate': 'not_evaluated', 'speed_comparison': 'not_evaluated'}
    def call(args, name, expected=0, prefix=wrapper):
        value = execute([*prefix, *map(str, args)], results, case['resources'], name, cg, expected)
        result['commands'][name] = value
    try:
        # Shell is used for a read-only namespace proof, never for the CLI argv.
        script = 'test -z "${VIRTUAL_ENV+x}" && test -z "${PYTHONPATH+x}" && test "$PATH" = /usr/bin:/bin && readlink /proc/self/ns/net'
        for index, root in enumerate(case['hidden_paths'], 1):
            script += f' && test -d "${index}" && test "$(stat -f -c %T -- "${index}")" = tmpfs && test -z "$(ls -A -- "${index}")"'
        call(['/bin/sh', '-c', script, 'isolation-proof', *case['hidden_paths']], 'isolation')
        namespace = (results / 'isolation.log').read_text().splitlines()[0]
        if namespace == os.readlink('/proc/self/ns/net') or not re.fullmatch(r'net:\[\d+\]', namespace):
            raise ValueError('Network namespace isolation was not observed')
        result.update(network_disabled=True, source_hidden=True, network_namespace=namespace,
                      python_environment='cleared; inference executes only native CLI', model_relocation='read-only bind at a Chinese-space path; original checkout hidden')
        call([binary, '--help'], 'help')
        call([binary, '--diagnose'], 'diagnose')
        if 'vulkan_compiled=true' not in (results / 'diagnose.log').read_text():
            raise ValueError('Expected a Vulkan-enabled installed runtime')
        call([binary, '--model', output / '模型 shared', '--verify-model'], 'verify-model')
        call([binary, '--diagnose'], 'no-gpu-diagnose',
             prefix=sandbox(output, model, case['hidden_paths'], no_gpu=True))
        if 'gpu_count=0' not in (results / 'no-gpu-diagnose.log').read_text().splitlines():
            raise ValueError('Explicit unavailable-driver diagnostic did not report zero GPUs')
        call([binary, '--model', results / '缺失 model', '--verify-model'], 'missing-model', 1)
        request = case['request']
        argv = [binary, '--model', output / '模型 shared', '--output', results / '生成 image.png',
                '--width', request['width'], '--height', request['height'], '--steps', 8,
                '--device', 'vulkan', '--precision', 'fp32', '--vae-device', 'cpu',
                '--vae-convolution', 'direct', '--text-device', 'cpu', '--threads', 2]
        for name, flag in [('prompt', '--prompt-file'), ('latent', '--latent'), ('image', '--input')]:
            if name in inputs:
                argv += [flag, inputs[name]['path']]
        if request['kind'] == 'img2img':
            argv += ['--resize', 'stretch', '--strength', request['strength']]
        if request['text_down_vector']:
            argv += ['--text-down-vector']
        call(argv, 'generate')
        png = results / '生成 image.png'
        png_container(png, request['width'], request['height'])
        result['output'] = identity(png)
        result['byte_identical_to_frozen_native_png'] = digest(png) == case['expected_png']['sha256']
        if not result['byte_identical_to_frozen_native_png']:
            raise ValueError('Offline output differs from the frozen native PNG; investigate without loosening this check')
        # Argument validation must refuse overwriting before loading a model again.
        call(argv, 'existing-output', 1)
        if digest(png) != result['output']['sha256']:
            raise ValueError('Existing output was modified')
        call([binary, '--model', output / '模型 shared', '--verify-model'], 'verify-model-after')
        checked_record(case['model']['manifest'])
        for name, expected in prep['files'].items():
            if digest(output / name) != expected:
                raise ValueError('Prepared runtime/input changed during execution')
        events = dict(line.split() for line in (cg / 'memory.events').read_text().splitlines())
        if any(int(events.get(k, 0)) for k in ('oom', 'oom_kill', 'oom_group_kill')):
            raise ValueError('The smoke scope has OOM events')
        result['status'] = 'passed_fixed_development_case'
    except BaseException as error:
        result.update(status='failed', error=str(error))
        raise
    finally:
        (output / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path, metavar='CASE_JSON')
    group.add_argument('--run', type=Path, metavar='PREPARED_DIRECTORY')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.prepare:
            if not args.output:
                parser.error('--prepare requires --output')
            value = prepare(args.prepare, args.output)
        else:
            if args.output:
                parser.error('--run does not accept --output')
            value = run(args.run)
        print(json.dumps({'status': value['status'], 'distributable': False}))
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as error:
        parser.exit(1, f'Offline release check failed: {error}\n')


if __name__ == '__main__':
    main()
