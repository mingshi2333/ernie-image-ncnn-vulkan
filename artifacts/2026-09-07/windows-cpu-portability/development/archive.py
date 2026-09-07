"""Archive small Windows portability evidence; never copy weights or Wine profiles."""
from pathlib import Path
import datetime, hashlib, json, os, re, shutil, subprocess, sys

ROOT = Path.cwd()
OUT = ROOT / 'outputs/windows-cpu-v1'
DEST = ROOT / 'artifacts/2026-09-07/windows-cpu-portability'
DEST.mkdir(exist_ok=False)
sys.path.insert(0, str(ROOT / 'tools'))
from source_inventory import source_files

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def write(name, value):
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def copy(path, name):
    assert path.is_file() and not path.is_symlink() and path.stat().st_size < 2_000_000, path
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)

def run(argv, env=None):
    p = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=20, env=env)
    return {'argv': list(map(str, argv)), 'return_code': p.returncode, 'output': p.stdout + p.stderr}

for f in sorted(OUT.rglob('*')):
    if f.is_file() and not f.is_symlink() and f.stat().st_size < 2_000_000:
        copy(f, Path('development') / f.relative_to(OUT))
for version in range(1, 5):
    base = Path(f'/var/tmp/ernie-windows-sdk-v{version}')
    for pattern in ('*.json', '*.log', '*.xml', 'toolchain.cmake', 'consumer/*.cpp', 'consumer/CMakeLists.txt',
                    'consumer-build/compile_commands.json', 'consumer-build/CMakeCache.txt',
                    'consumer-build/CMakeFiles/4.3.0/CMakeCXXCompiler.cmake',
                    '移动 installation/lib/cmake/*/*.cmake'):
        for f in sorted(base.glob(pattern)): copy(f, Path(f'sdk-v{version}') / f.relative_to(base))
base = Path('/var/tmp/ernie-windows-native-links-v1')
for pattern in ('*.json', '*.log'):
    for f in sorted(base.glob(pattern)): copy(f, Path('native-links') / f.name)

sources = [{'path': str(f.relative_to(ROOT)), 'bytes': f.stat().st_size, 'sha256': digest(f)}
           for f in source_files(ROOT)]
write('source-inventory.json', {'base_commit': run(['git', 'rev-parse', 'HEAD'])['output'].strip(),
      'scope': 'Post-UTF8 production/build/test source; documentation is not part of this inventory',
      'files': sources})
diff = subprocess.check_output(['git', 'diff', '--binary', '--', 'CMakeLists.txt', 'include', 'cli', 'src',
                               'cmake', 'probes', 'tests', 'tokenizer', 'tools'])
for name in ('cli/windows_main.cpp', 'tests/test_utf8_paths.cpp'):
    p = subprocess.run(['git', 'diff', '--no-index', '--binary', '--', '/dev/null', name], capture_output=True)
    assert p.returncode == 1
    diff += p.stdout
(DEST / 'source.patch').write_bytes(diff)

builds = {}
for name in ('build-windows-cpu-v1', 'build-install-cpu', 'build-dev'):
    directory = ROOT / name
    files = sorted(directory.glob('*.exe')) if name == 'build-windows-cpu-v1' else [directory / 'ernie-image']
    files += sorted(directory.glob('src/*.a')) + sorted(directory.glob('ncnn/src/*.a'))
    entries = [{'path': str(f.relative_to(ROOT)), 'bytes': f.stat().st_size, 'sha256': digest(f)} for f in files if f.is_file()]
    config = {}
    for line in (directory / 'CMakeCache.txt').read_text().splitlines():
        if re.match(r'(ERNIE_|NCNN_|CMAKE_BUILD_TYPE:|CMAKE_CXX_COMPILER:|CMAKE_C_COMPILER:|CMAKE_TOOLCHAIN_FILE:|CARGO_EXECUTABLE:|OpenMP_)', line) and '=' in line:
            k, value = line.split('=', 1); config[k] = value
    builds[name] = {'files': entries, 'cmake_selected': config,
                   'scope': 'Final UTF8 code' if name != 'build-dev' else 'Unchanged older Linux Vulkan build; not rebuilt with Windows changes'}
write('final-builds.json', builds)

closure = json.loads((OUT / 'windows-imports.json').read_text())
imports = {}
for entry in builds['build-windows-cpu-v1']['files']:
    f = ROOT / entry['path']
    if f.suffix != '.exe': continue
    result = run(['/usr/bin/x86_64-w64-mingw32-objdump', '-p', f])
    assert result['return_code'] == 0
    imports[entry['path']] = re.findall(r'DLL Name: (\S+)', result['output'])
write('final-executable-imports.json', {'imports': imports,
      'runtime_dll_closure': 'development/windows-imports.json; actual copied DLL hashes also in sdk-v4/result.json',
      'scope': 'Import inspection and actual Wine execution, not DLL redistribution approval'})

toolchain = Path('/home/mingshi/.rustup/toolchains/1.98.0-x86_64-unknown-linux-gnu/bin')
commands = [['/usr/bin/x86_64-w64-mingw32-g++', '--version'], [str(toolchain / 'rustc'), '-Vv'],
            [str(toolchain / 'cargo'), '--version'], ['/usr/bin/wine', '--version'], ['cmake', '--version'],
            ['git', '-C', str(ROOT.parent.parent / 'third_party/ncnn'), 'rev-parse', 'HEAD'],
            ['git', '-C', str(ROOT.parent.parent / 'third_party/ncnn'), 'status', '--porcelain']]
write('toolchain-and-upstream.json', {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'commands': [run(command) for command in commands]})
env = os.environ.copy()
env.update(XDG_RUNTIME_DIR='/run/user/1000', DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
units = ['ernie-windows-utf8-tests-v1', 'ernie-windows-utf8-build-v1', 'ernie-linux-utf8-build-v1',
         'ernie-linux-utf8-tests-v1', 'ernie-windows-sdk-v3', 'ernie-windows-sdk-v4']
write('unit-status.json', {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'units': [run(
    ['systemctl', '--user', 'show', u, '-p', 'LoadState,ActiveState,SubState,Result,ExecMainCode,ExecMainStatus,MemoryPeak,MemorySwapPeak,CPUUsageNSec,ControlGroup,KillSignal,TimeoutStopUSec'], env)
    for u in units]})
journal = run(['journalctl', '--user-unit=ernie-windows-sdk-v3.service', '--no-pager', '-n', '35'], env)
write('sdk-v3-cleanup-journal.json', journal)
write('collection-notes.json', {
    'link_probe': 'development/link-semantics.rs is the final extended helper used by native-links/result.json. The earlier metadata-only link-semantics-results.json did not freeze its initial helper source. Its raw result remains historical; no reconstructed source hash is claimed.',
    'source_bindings': 'source-inventory and source.patch describe final code. Pre-UTF8 binaries and the six-file patch are separately retained. Earlier build iterations predate this source identity.',
    'installed_docs': 'SDK records bind the actual documentation installed during each run; later explanatory documentation edits do not rebind those old records.',
    'unit_status': 'Some successful transient units may already be unloaded; terminal output is separately recorded where available.'})
print(json.dumps({'directory': str(DEST), 'source_files': len(sources), 'records': sum(f.is_file() for f in DEST.rglob('*')),
                  'windows_cli': next(e for e in builds['build-windows-cpu-v1']['files'] if e['path'].endswith('/ernie-image.exe'))}, indent=2))
