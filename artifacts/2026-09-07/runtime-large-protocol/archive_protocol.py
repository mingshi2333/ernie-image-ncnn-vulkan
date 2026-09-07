"""Archive only small, frozen preparation evidence before any model run."""
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path(__file__).resolve().parent
WORK=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
DEST=WORK/'artifacts/2026-09-07/runtime-large-protocol'
def identity(path):
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'bytes':path.stat().st_size,'sha256':digest}
assert not (BASE/'progress.json').exists()
for name,item in json.loads((BASE/'batch-identity.json').read_text()).items():assert identity(Path(name))==item
paths=[BASE/name for name in ('prepare.py','run_batch.py','test_batch.py','policy-checks.json','policy-checks.log',
    'prepare.log','preflight.py','preflight.json','preflight.log','native-lineage.json','native-lineage.patch',
    'commands.json','batch-identity.json','build-identity.json','CMakeCache.snapshot.txt','archive_protocol.py')]
for label in ('1376x768','768x1376','2048x1024','1024x2048'):
    paths.extend(BASE/label/name for name in ('plan.json','request.json','worker.py','run.py','supervisor.py','compare.py'))
for path in paths:
    destination=DEST/path.relative_to(BASE);destination.parent.mkdir(parents=True,exist_ok=True)
    assert not destination.exists();shutil.copyfile(path,destination)
    assert identity(path)==identity(destination)
retained=[BASE/'ernie-image.snapshot',*(BASE/label/'initial.f32' for label in ('1376x768','768x1376','2048x1024','1024x2048'))]
retained.extend(p for p in (BASE/'source').rglob('*') if p.is_file() and '/models/' not in str(p))
with (DEST/'retained-local-files.json').open('x') as stream:
    stream.write(json.dumps({str(p):identity(p) for p in sorted(retained)},indent=2)+'\n')
inventory={str(p.relative_to(DEST)):identity(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
with (DEST/'inventory.json').open('x') as stream:stream.write(json.dumps(inventory,indent=2)+'\n')
print(json.dumps({'copied_files':len(paths),'inventory_files':len(inventory),'retained_files':len(retained),'passed':True}))
