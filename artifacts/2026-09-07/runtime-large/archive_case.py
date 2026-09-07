"""Keep small completed-case evidence in Git; retain tensors and sampling locally."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

BASE=Path(__file__).resolve().parent
WORK=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference')
assert len(sys.argv)==2 and sys.argv[1] in ('1376x768','768x1376','2048x1024','1024x2048')
label=sys.argv[1];case=BASE/label
DEST=WORK/'artifacts/2026-09-07/runtime-large'/label
def identity(path):
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'bytes':path.stat().st_size,'sha256':digest}
review=json.loads((case/'root-review.json').read_text())
assert review['complete_execution_both'] and review['verified_bindings']==436
assert review['reviewer_sha256']==identity(BASE/'review_case.py')['sha256']
plan=json.loads((case/'plan.json').read_text())
for name,item in plan['bindings'].items():assert identity(Path(name))==item,name
paths=[case/name for name in ('request.json','comparison.json','root-review.json','registry-validation.json')]
for phase in ('native','official'):
    process=json.loads((case/phase/'process.json').read_text())
    assert process['complete'] and process['return_code']==0
    paths.extend(case/phase/name for name in ('process.json','worker-result.json','runner.log','command-0.log'))
paths.extend((case/'native/report.json',case/'official/reference/fixture.json'))
paths.extend((case/'native/trace').glob('*.txt'))
DEST.mkdir()
for path in sorted(paths):
    destination=DEST/path.relative_to(case);destination.parent.mkdir(parents=True,exist_ok=True)
    assert path.stat().st_size<=4*1024*1024,path
    shutil.copyfile(path,destination);assert identity(path)==identity(destination)
for suffix in ('native.log','official.log','compare.log','review.log','register.log'):
    path=BASE/(label+'-'+suffix);destination=DEST/('batch-'+suffix)
    shutil.copyfile(path,destination);assert identity(path)==identity(destination)
retained=[p for p in case.rglob('*') if p.is_file() and not p.is_symlink() and p not in paths]
with (DEST/'retained-local-files.json').open('x') as out:
    out.write(json.dumps({str(p):identity(p) for p in sorted(retained)},indent=2)+'\n')
inventory={str(p.relative_to(DEST)):identity(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
with (DEST/'inventory.json').open('x') as out:out.write(json.dumps(inventory,indent=2)+'\n')
print(json.dumps({'case':label,'inventory_files':len(inventory),'retained_files':len(retained),'passed':True}),flush=True)
