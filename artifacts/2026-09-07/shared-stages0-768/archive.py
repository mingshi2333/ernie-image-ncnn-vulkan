import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
BASE = Path('/var/tmp/ernie-stages0-768-v1')
DEST = ROOT/'artifacts/2026-09-07/shared-stages0-768'

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def identity(path):
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}

def write(name, data):
    (DEST/name).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')

plan = json.loads((BASE/'plan.json').read_text())
sources = json.loads((BASE/'source-before.json').read_text())
audit = json.loads((BASE/'audit.json').read_text())
assert audit['complete_execution'] and audit['official_prediction_exact_replay'] and audit['native_prediction_exact_replay']
assert audit['stage_count'] == 44 and audit['verified_bindings'] == 419
assert digest(BASE/'plan.json') == audit['plan_sha256']
assert len(sources) == 336
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == plan['source_head']
DEST.mkdir(parents=True, exist_ok=True)
assert not (DEST/'archive-check.json').exists(), 'Do not overwrite an existing completed archive'
selected = [
    'prepare.py', 'prepare.log', 'plan.json', 'source-before.json', 'run.py', 'supervisor.py',
    'supervisor.log', 'tests.json', 'tests.log', 'audit_stages.py', 'audit.log', 'audit.json',
    'archive.py', 'native/process.json', 'native/worker-result.json', 'native/command-0.log',
    'native/runner.log', 'native/diagnostic/result.json', 'native/diagnostic/oracle.log',
    'native/diagnostic/oracle/fixture/fixture.json', 'native/diagnostic/native/result.json',
    'native/diagnostic/native/runner.log', 'native/diagnostic/native/resources.log',
    'native/diagnostic/native/gpu-device-memory.log',
]
copied = {}
for relative in selected:
    target = DEST/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASE/relative, target)
    assert identity(target) == identity(BASE/relative)
    copied[relative] = identity(target)

tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
existing = sorted(set(sources) & tracked)
added = sorted(set(sources) - tracked)
assert added == ['tests/test_diagnose_dit_stages.py']
patch = subprocess.check_output(['git', 'diff', '--binary', plan['source_head'], '--', *existing], cwd=ROOT)
for relative in added:
    result = subprocess.run(['git', 'diff', '--no-index', '--binary', '--', '/dev/null', relative], cwd=ROOT, capture_output=True)
    assert result.returncode == 1 and not result.stderr
    patch += result.stdout
(DEST/'source.patch').write_bytes(patch)
with tempfile.TemporaryDirectory(prefix='ernie-stages0-source-') as directory:
    rebuilt = Path(directory)
    original = subprocess.check_output(['git', 'archive', '--format=tar', plan['source_head'], '--', *existing], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(original)) as archive:
        archive.extractall(rebuilt, filter='data')
    subprocess.run(['git', 'apply', '--check', str(DEST/'source.patch')], cwd=rebuilt, check=True)
    subprocess.run(['git', 'apply', str(DEST/'source.patch')], cwd=rebuilt, check=True)
    for relative, expected in sources.items():
        for path in (ROOT/relative, BASE/'source'/relative, rebuilt/relative):
            assert identity(path) == expected, str(path)

scripts = BASE/'native/diagnostic/scripts'
script_files = sorted(scripts.glob('*.py'))
assert len(script_files) == 112
for path in script_files:
    assert digest(path) == digest(BASE/'source/tools'/path.name), str(path)
for name, expected in plan['bindings'].items():
    assert identity(Path(name)) == expected, name
assert digest(BASE/'runner.snapshot') == plan['runner_sha256']
assert digest(BASE/'native/diagnostic/runner.snapshot') == plan['runner_sha256']
assert json.loads((BASE/'tests.json').read_text())['return_code'] == 0
assert 'Ran 25 tests' in (BASE/'tests.log').read_text()

retained = sorted(set(BASE.rglob('*.f32')) | {BASE/'runner.snapshot', BASE/'native/diagnostic/runner.snapshot', BASE/'native/samples.jsonl'})
assert all('/source/' not in str(p) and '/scripts/' not in str(p) for p in retained)
write('retained-files.json', {str(p): identity(p) for p in retained})
ncnn = ROOT.parents[1]/'third_party/ncnn'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ncnn, text=True).strip() == '6a1bf000f363714839a36793addc8c879d3d899e'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ncnn, text=True).strip()
write('archive-check.json', {
    'source_head': plan['source_head'], 'source_files_reconstructed_and_current': len(sources),
    'source_patch_sha256': digest(DEST/'source.patch'), 'copied_small_files': copied,
    'copied_python_sources_verified': len(script_files), 'retained_local_files': len(retained),
    'plan_sha256': digest(BASE/'plan.json'), 'verified_bindings': len(plan['bindings']),
    'runner_sha256': plan['runner_sha256'], 'related_tests_passed': 25,
    'inference_math_changed': False, 'native_acceptance_eligible': False,
    'ncnn_clean_revision': '6a1bf000f363714839a36793addc8c879d3d899e',
    'full_768_tensor_result_remains': '18/25',
})
print(json.dumps({'copied_small_files': len(copied), 'source_files':len(sources),
                  'retained_files':len(retained), 'bindings':len(plan['bindings']), 'passed':True}))
