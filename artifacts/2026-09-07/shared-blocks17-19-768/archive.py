"""Retain the selected-block evidence without copying tensors or weights into Git."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

B=Path(__file__).resolve().parent
R=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
A=R/'artifacts/2026-09-07/shared-blocks17-19-768'
def identity(path):
    path=Path(path)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'bytes':path.stat().st_size,'sha256':digest}
def write(name,value):
    (A/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
plan=json.loads((B/'plan.json').read_text())
audit=json.loads((B/'audit.json').read_text())
sources=json.loads((B/'source-before.json').read_text())
assert audit['complete_execution'] and audit['official_exact_replays']==audit['native_exact_replays']==3
assert identity(B/'plan.json')['sha256']==audit['plan_sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==plan['source_head']
assert not (A/'archive-check.json').exists()
selected=['experiment.py','plan.json','source-before.json','run.py','supervisor.py','build-record.json',
          'cli-checks.json','position-preparation.json','preparation-record.json','official-supervisor.log',
          'native-supervisor.log','audit.py','audit.log','audit.json','archive.py']
for phase in ('official','native'):
    selected.extend(f'{phase}/{name}' for name in ('process.json','worker-result.json','command-0.log','runner.log'))
selected.extend(['official/results/result.json','native/results.json'])
for case in plan['cases']:
    directory=f"native/block-{case['block']}-{case['variant']}"
    selected.extend(directory+'/'+name for name in ('runner.log','resources.log'))
for relative in selected:
    destination=A/relative;destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(B/relative,destination)
    assert identity(destination)==identity(B/relative)
changed=subprocess.check_output(['git','diff','--name-only',plan['source_head'],'--',*sorted(sources)],cwd=R,text=True).splitlines()
assert changed==['probes/block_sequence_runner.cpp']
patch=subprocess.check_output(['git','diff','--binary',plan['source_head'],'--',*sorted(sources)],cwd=R)
(A/'source.patch').write_bytes(patch)
with tempfile.TemporaryDirectory(prefix='ernie-blocks17-19-source-') as directory:
    target=Path(directory)
    original=subprocess.check_output(['git','archive','--format=tar',plan['source_head'],'--',*sorted(sources)],cwd=R)
    with tarfile.open(fileobj=io.BytesIO(original)) as archive:archive.extractall(target,filter='data')
    subprocess.run(['git','apply','--check',str(A/'source.patch')],cwd=target,check=True)
    subprocess.run(['git','apply',str(A/'source.patch')],cwd=target,check=True)
    for relative,expected in sources.items():
        assert identity(target/relative)==identity(R/relative)==identity(B/'source'/relative)==expected,relative
retained=set(B.rglob('*.f32'))|{B/'runner.snapshot',B/'official/samples.jsonl',B/'native/samples.jsonl'}
for case in plan['cases']:
    directory=B/f"native/block-{case['block']}-{case['variant']}"
    retained.update([directory/'result.json',directory/'fixture/fixture.json'])
write('retained-files.json',{str(path):identity(path) for path in sorted(retained)})
for path,expected in plan['bindings'].items():assert identity(path)==expected,path
write('archive-check.json',{'source_head':plan['source_head'],'source_files_reconstructed_and_current':len(sources),
    'source_patch':identity(A/'source.patch'),'copied_small_files':len(selected),'retained_files':len(retained),
    'plan_sha256':identity(B/'plan.json')['sha256'],'verified_bindings':len(plan['bindings']),
    'inference_source_changed':False,'probe_changed':True,'native_acceptance_eligible':False,
    'official_replays':3,'native_replays':3,'native_calls':9,'all_model_processes_complete':True})
inventory={path.relative_to(A).as_posix():identity(path) for path in sorted(A.rglob('*')) if path.is_file() and path.name!='inventory.json'}
write('inventory.json',inventory)
assert all(identity(A/name)==expected for name,expected in json.loads((A/'inventory.json').read_text()).items())
print(json.dumps({'archive_files':len(inventory)+1,'copied_small_files':len(selected),'retained_files':len(retained),
                  'verified_bindings':len(plan['bindings']),'reconstructed_sources':len(sources),'passed':True}))
