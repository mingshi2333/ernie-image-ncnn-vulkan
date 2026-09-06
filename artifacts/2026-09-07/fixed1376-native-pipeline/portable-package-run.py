import json
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
ROOT = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
sys.path.insert(0, str(BASE / 'source/tools'))
from package_model import package_model, verify_package, sha256

start = time.monotonic()
original = ROOT / 'models/turbo1024-s64-portable'
source_sha = sha256(original / 'manifest.json')
output = ROOT / 'models/turbo1376x768-s64-portable'
manifest = package_model(original, output, link=False, fixed1376=True)
verify_package(output)
assert not any(path.is_symlink() for path in output.rglob('*'))
executed = json.loads((ROOT / 'outputs/f1-pipeline1376-v1/model/manifest.json').read_text())
assert manifest['files'] == executed['files'] and manifest['file_sizes'] == executed['file_sizes']
assert manifest['config'] == executed['config']
assert manifest['source_weights'] == executed['source_weights']
assert sha256(original / 'manifest.json') == source_sha
sources = json.loads((BASE / 'source-identity.json').read_text())
assert all(sha256(BASE / 'source' / name) == digest for name, digest in sources.items())
runner = ROOT / 'outputs/f1-pipeline1376-v1/ernie-image.snapshot'
with (BASE / 'native-verify.log').open('w') as log:
    result = subprocess.run([str(runner), '--model', str(output), '--verify-model'], stdout=log, stderr=subprocess.STDOUT)
report = {'complete': result.returncode == 0, 'native_return_code': result.returncode,
          'package_manifest_sha256': sha256(output / 'manifest.json'),
          'executed_package_manifest_sha256': sha256(ROOT / 'outputs/f1-pipeline1376-v1/model/manifest.json'),
          'runtime_files_byte_identical': len(manifest['files']), 'config_identical': True,
          'official_weight_bindings_identical': True, 'portable': True,
          'source_manifest_sha256_before_and_after': source_sha,
          'source_files_verified': len(sources), 'source_identity_sha256': sha256(BASE / 'source-identity.json'),
          'runner_sha256': sha256(runner), 'wall_seconds': time.monotonic() - start}
(BASE / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
raise SystemExit(result.returncode)
