import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path.cwd(); source=(root/'outputs/d1-preset-source-v5').resolve()
out=root/'outputs/d1-preset-vulkan-v5';out.mkdir(exist_ok=False)
metadata=source/'source-identity.json';f=json.loads(metadata.read_text())
def changed():
    return [p for p,sha in f['files'].items() if hashlib.sha256((source/p).read_bytes()).hexdigest()!=sha]
assert not changed()
ncnn=Path('/home/mingshi/Project/AI/ernie-image-ncnn-vulkan/third_party/ncnn')
assert (ncnn/'CMakeLists.txt').is_file()
ncnn_head=subprocess.check_output(['git','-C',str(ncnn),'rev-parse','HEAD'],text=True).strip()
assert ncnn_head=='6a1bf000f363714839a36793addc8c879d3d899e'
def dependency_changes():
    return subprocess.check_output(['git','-C',str(ncnn),'status','--porcelain','--untracked-files=no'],text=True)
assert not dependency_changes()
commands=[['cmake','--preset','linux-vulkan','-DERNIE_NCNN_SOURCE_DIR='+str(ncnn)], ['cmake','--build','--preset','linux-vulkan']]
identity=dict(commands=commands,source_identity_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),ncnn_revision=ncnn_head,dependencies=subprocess.check_output(['git','-C',str(ncnn),'submodule','status','--recursive'],text=True),resource_controls={})
line=next(x for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
cgroup=Path('/sys/fs/cgroup')/line.split('::',1)[1].lstrip('/')
for name in ('memory.max','memory.swap.max','cpu.max'):
    identity['resource_controls'][name]=(cgroup/name).read_text().strip()
identity['cpu_affinity']=sorted(os.sched_getaffinity(0))
(out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
results=[];exit_code=0
for name,command in zip(('configure','build','test'),commands):
    start=time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT)
    results.append(dict(name=name,exit_code=result.returncode,wall_seconds=time.monotonic()-start))
    print(name,result.returncode,flush=True)
    if result.returncode:
        exit_code=result.returncode;break
changes=changed();dirty=dependency_changes()
(out/'result.json').write_text(json.dumps(dict(commands=results,exit_code=exit_code,source_changes_after=changes,ncnn_tracked_changes_after=dirty),indent=2)+'\n')
raise SystemExit(exit_code or bool(changes) or bool(dirty))
