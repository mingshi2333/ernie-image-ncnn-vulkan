import hashlib
import json
from pathlib import Path
import sys
import unittest

workspace = Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
base = Path(__file__).resolve().parent
sys.path[:0] = [str(workspace / 'tools'), str(workspace / 'tests')]
from pipeline_package import select_shared_instance
from pipeline_reference import full_reference_contract, reviewed_shared_reference

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

source_paths = [workspace / 'tools' / name for name in
                ('pipeline_package.py', 'pipeline_reference.py')]
source_before = {str(path): sha(path) for path in source_paths}
manifest_path = workspace / 'models/turbo-shared-v2/manifest.json'
manifest = json.loads(manifest_path.read_text())
assert sha(manifest_path) == '21b6bc8deae17418c22257b52e928ee048a372342a2517bcb954276257c21d1a'
registered = []
for side in (768, 1024):
    fixture_path = base / str(side) / 'official/reference/fixture.json'
    fixture = json.loads(fixture_path.read_text())
    selected, target = select_shared_instance(manifest['instances'], side, side, len(fixture['ids']))
    binding = dict(schema_version=3,
                   source_manifest_sha256=selected['source_manifest_sha256'],
                   shared_manifest_sha256=sha(manifest_path),
                   runtime_bindings=selected['runtime_bindings'])
    if target is not None:
        binding['runtime_target'] = target
    reviewed = reviewed_shared_reference(fixture_path, binding, manifest_path)
    count = full_reference_contract(fixture, selected['config'], fixture['prompt'], 8)
    assert count == 25
    registered.append(dict(side=side, authenticated=True, complete_boundaries=count,
                           reference=str(fixture_path), reviewed=reviewed))

suite = unittest.defaultTestLoader.loadTestsFromNames(
    ['test_pipeline_reference', 'test_pipeline_package'])
result = unittest.TextTestRunner(verbosity=2).run(suite)
source_after = {str(path): sha(path) for path in source_paths}
assert source_before == source_after
record = dict(scope='Current registry authenticates both complete official fixtures, source manifest objects, runtime bindings and target configurations; this check does not rehash all model weight objects or rerun inference.',
              registered=registered, tests_run=result.testsRun,
              failures=len(result.failures), errors=len(result.errors),
              successful=result.wasSuccessful(), source_sha256=source_after)
(base / 'registry-validation.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
raise SystemExit(0 if result.wasSuccessful() else 1)
