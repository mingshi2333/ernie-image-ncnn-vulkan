"""Authenticate a completed reviewed fixture through the current public registry."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

WORK=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
BASE=Path(__file__).resolve().parent
assert len(sys.argv)==2 and sys.argv[1] in ('1376x768','768x1376','2048x1024','1024x2048')
case=BASE/sys.argv[1]
sys.path[:0]=[str(WORK/'tools'),str(WORK/'tests')]
from pipeline_package import select_shared_instance
from pipeline_reference import full_reference_contract,reviewed_shared_reference
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
sources=[WORK/'tools'/name for name in ('pipeline_package.py','pipeline_reference.py')]
source_before={str(path):sha(path) for path in sources}
manifest_path=WORK/'models/turbo-shared-v2/manifest.json'
manifest=json.loads(manifest_path.read_text())
assert sha(manifest_path)=='21b6bc8deae17418c22257b52e928ee048a372342a2517bcb954276257c21d1a'
fixture_path=case/'official/reference/fixture.json';fixture=json.loads(fixture_path.read_text())
width,height=json.loads((case/'request.json').read_text())['runtime_size']
selected,target=select_shared_instance(manifest['instances'],width,height,len(fixture['ids']))
binding={'schema_version':3,'source_manifest_sha256':selected['source_manifest_sha256'],
         'shared_manifest_sha256':sha(manifest_path),'runtime_bindings':selected['runtime_bindings']}
if target is not None:binding['runtime_target']=target
reviewed=reviewed_shared_reference(fixture_path,binding,manifest_path)
assert full_reference_contract(fixture,selected['config'],fixture['prompt'],8)==25
review=json.loads((case/'root-review.json').read_text())
assert review['complete_execution_both'] and review['reference_fixture_sha256']==sha(fixture_path)
suite=unittest.defaultTestLoader.loadTestsFromNames(['test_pipeline_reference','test_pipeline_package'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
source_after={str(path):sha(path) for path in sources};assert source_before==source_after
record={'scope':'Current registry authentication, source-manifest object and full25-boundary contract; independent all-tensor/command readback precedes registration; this check does not rerun the model.',
        'case':case.name,'shape':[width,height],'reviewed':reviewed,'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'successful':result.wasSuccessful(),
        'source_sha256':source_after}
with (case/'registry-validation.json').open('x') as out:out.write(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
raise SystemExit(0 if result.wasSuccessful() else 1)
