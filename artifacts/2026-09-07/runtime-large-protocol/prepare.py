"""Freeze the four remaining normal-aspect large shared-source canaries."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan')
WORK=ROOT/'.worktrees/surpass-reference'
OLD=Path('/var/tmp/ernie-runtime-squares-v1')
SIZES=((1376,768),(768,1376),(2048,1024),(1024,2048))
sys.path.insert(0,str(WORK/'tools'))
from source_inventory import source_files

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def identity(path):
    path=Path(path);return {'bytes':path.stat().st_size,'sha256':sha(path)}
def save(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

assert not subprocess.check_output(['git','status','--porcelain'],cwd=WORK)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip()=='0443a6d070b637e872c9f8362806b15455d19c5e'
old_plan=json.loads((OLD/'768/plan.json').read_text())
for name,item in old_plan['bindings'].items():assert identity(name)==item,name
source=BASE/'source';source.mkdir()
for path in source_files(OLD/'source'):
    destination=source/path.relative_to(OLD/'source');destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,destination);destination.chmod(0o444)
(source/'models').symlink_to(ROOT/'models',target_is_directory=True)
for name in ('ernie-image.snapshot','CMakeCache.snapshot.txt','build-identity.json'):
    shutil.copyfile(OLD/name,BASE/name)
binary=BASE/'ernie-image.snapshot';binary.chmod(0o555)
assert sha(binary)==old_plan['binary_sha256']=='92559ea493adabfb1a0b2fbf1d69e21ccb79d7c9f2dbd37cb25898d7654da1c0'
build=json.loads((BASE/'build-identity.json').read_text())
assert build['source_files']==len(source_files(source))==334
ncnn=ROOT/'third_party/ncnn'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ncnn)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ncnn,text=True).strip()==build['ncnn_revision']
changed=subprocess.check_output(['git','diff','--name-only',build['source_head']+'..HEAD','--','src','include','cli'],cwd=WORK,text=True).splitlines()
assert changed==['src/host_memory.cpp','src/host_memory.h']
patch=subprocess.check_output(['git','diff',build['source_head']+'..HEAD','--','src','include','cli'],cwd=WORK)
(BASE/'native-lineage.patch').write_bytes(patch)
save(BASE/'native-lineage.json',{'source_head':build['source_head'],
    'preparation_head':'0443a6d070b637e872c9f8362806b15455d19c5e','binary_sha256':sha(binary),
    'current_native_changes':changed,'native_patch_sha256':sha(BASE/'native-lineage.patch'),
    'scope':'Reuse the completed 64/512/768/1024 Linux generator and official source snapshot; changes since its source are Windows memory-reader code and header comments; this is the frozen 990e8ef generator, not a current-HEAD build.'})
policy=json.loads((BASE/'policy-checks.json').read_text());assert policy['passed'] and policy['tests']==13
common_files=[*source_files(source),binary,*(BASE/name for name in
    ('CMakeCache.snapshot.txt','build-identity.json','prepare.py','run_batch.py','test_batch.py',
     'policy-checks.json','policy-checks.log','native-lineage.json','native-lineage.patch'))]
common_files.extend(Path(name) for name in old_plan['bindings'] if '/models/' in name and '/source/' not in name)
common={str(p):identity(p) for p in sorted(set(common_files))}
shared=ROOT/'models/turbo-shared-v2'
assert common[str(shared/'manifest.json')]['sha256']==old_plan['model_manifest_sha256']
old_request=json.loads((OLD/'768/request.json').read_text())
commands=[]
def replace_paths(value,case):
    if isinstance(value,list):return [replace_paths(x,case) for x in value]
    if isinstance(value,str):
        return value.replace(str(OLD/'768'),str(case)).replace(str(OLD/'ernie-image.snapshot'),str(binary))
    raise TypeError(type(value))
for width,height in SIZES:
    label=f'{width}x{height}';case=BASE/label;case.mkdir()
    (case/'source').symlink_to(source,target_is_directory=True)
    for name in ('model','temporary','cache'):(case/name).mkdir()
    initial=case/'initial.f32'
    np.random.Generator(np.random.PCG64(20260905)).standard_normal(
        (1,128,height//16,width//16),dtype=np.float32).astype('<f4').tofile(initial)
    request={**old_request,'runtime_size':[width,height],'initial':str(initial)}
    save(case/'request.json',request)
    for name in ('worker.py','run.py'):shutil.copyfile(OLD/'768'/name,case/name)
    supervisor=(OLD/'768/supervisor.py').read_text()
    assert supervisor.count("'ernie-square-768-v1-'")==1
    (case/'supervisor.py').write_text(supervisor.replace("'ernie-square-768-v1-'",f"'ernie-large-{label}-v1-'"))
    comparator=(OLD/'768/compare.py').read_text();assert comparator.count('runtime768x768')==1
    (case/'compare.py').write_text(comparator.replace('runtime768x768','runtime'+label))
    bindings=dict(common)
    for name in ('initial.f32','request.json','worker.py','run.py','supervisor.py','compare.py'):
        path=case/name;path.chmod(0o444);bindings[str(path)]=identity(path)
    native=replace_paths(old_plan['commands']['native'][0],case)
    native[native.index('--width')+1]=str(width);native[native.index('--height')+1]=str(height)
    official=replace_paths(old_plan['commands']['official'][0],case)
    plan={**{k:v for k,v in old_plan.items() if k not in ('bindings','commands','scope')},
        'scope':f'Development full runtime{label} native/official FP32 canary with saved identical noise, source32 and 64 DiT text slots, 8 steps, explicit RAM weights/stdio/cache0, source hidden and network disabled. Frozen Linux source990e8ef/binary92559ea4. Not peer performance, dynamic auto-pressure, OOM recovery, or broad quality acceptance.',
        'commands':{'native':[native],'official':[official]},'bindings':bindings}
    save(case/'plan.json',plan);(case/'plan.json').chmod(0o444)
    for phase in ('native','official','compare'):
        argv=[str(WORK/'.venv/bin/python'),str(case/('compare.py' if phase=='compare' else 'supervisor.py'))]
        if phase!='compare':argv.append(phase)
        else:
            argv=['systemd-run','--user','--wait','--pipe','--unit=ernie-large-'+label+'-compare-v1',
                  '--property=AllowedCPUs=12,14','--property=CPUQuota=200%',
                  '--property=MemoryMax=4G','--property=MemorySwapMax=0',
                  '/usr/bin/env','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','PYTHONDONTWRITEBYTECODE=1',*argv]
        commands.append({'case':label,'phase':phase,'argv':argv})
    print(json.dumps({'case':label,'plan_sha256':sha(case/'plan.json'),'bindings':len(bindings),'initial_sha256':sha(initial)}),flush=True)
save(BASE/'commands.json',commands)
save(BASE/'batch-identity.json',{str(p):identity(p) for p in
    (BASE/'commands.json',BASE/'run_batch.py',*(BASE/f'{w}x{h}'/'plan.json' for w,h in SIZES))})
